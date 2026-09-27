"""GRPO for the one-pass letter reader (single-token policy, no decoding).

Policy: the softmax over the k option letters at the final prompt position -- identical to
inference. For each item we sample G actions from that distribution in one forward pass.

Reward (per sampled action a):
    R = 1[a == answer] + lambda * pi_perm(a)
where pi_perm(a) is the probability the same model assigns to the *same option* under a
permuted option order (label-free order-consistency reward: actions the model likes under
both orders are reinforced; constant-per-item terms cancel in the group advantage).

Loss: group-normalized advantage policy gradient + optional KL to the base policy
(computed by disabling the LoRA adapter), i.e. standard GRPO without a critic.
"""

import argparse
import io
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import bitsandbytes as bnb
from cauldron_tasks import LETTERS
from extract_cauldron import letter_ids, prepare
from train_qlora import ItemDataset, eval_all, load_pool, move, parse_sizes
from train_rft import build_model

MASK = -1e9


class RawCollate:
    def __call__(self, batch):
        return batch


def softmax_letters(logits_row, letters, k):
    return torch.log_softmax(logits_row[letters[:k]].float(), dim=-1)


def policy_forward(model, processor, batch, letters, args):
    """Return list of per-item log-softmax over the k letter options."""
    inputs = prepare(processor, batch, args.image_size, args.pad_multiple)
    logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :].float().cpu()
    return [softmax_letters(logits[i], letters, len(batch[i]["options"])) for i in range(len(batch))], inputs


CONF_TEXT = ("\nConfidence that the answer is correct "
             "(A=50%, B=55%, C=60%, D=65%, E=70%, F=75%, G=80%, H=85%, I=90%, J=95%, K=100%):")


def confidence_ids(letters):
    """A-K letter tokens as confidence buckets 50%..100% (each must be one token)."""
    ids = letters[:11]
    vals = [0.50 + 0.05 * i for i in range(11)]
    return ids, vals


def conf_forward(model, processor, batch, chosen_letters, conf_ids, args):
    """One pass per item: prompt + 'Answer: <letter>' + confidence scale -> logits for confidence."""
    from extract_cauldron import letterbox, prompt_text
    from qwen_vl_utils import process_vision_info
    messages = []
    for it, L in zip(batch, chosen_letters):
        text = prompt_text(it) + f"\nAnswer: {L}" + CONF_TEXT
        content = [{"type": "image", "image": letterbox(img, args.image_size)} for img in it["images"]]
        content.append({"type": "text", "text": text})
        messages.append([{"role": "user", "content": content}])
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for m in messages]
    imgs, _ = process_vision_info(messages)
    inputs = processor(text=texts, images=imgs, return_tensors="pt", padding=True,
                       pad_to_multiple_of=args.pad_multiple)
    logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :].float()
    return torch.log_softmax(logits[:, conf_ids], dim=-1).cpu()


def proper_score(c, y, kind):
    """Strictly proper reward for a stated confidence c and correctness y in {0,1}."""
    if kind == "brier":
        return 1.0 - (c - y) ** 2
    c = min(max(c, 1e-3), 1 - 1e-3)
    return (y * math.log(c) + (1 - y) * math.log(1 - c) + math.log(2)) / math.log(2)


def confidence_loss(model, processor, batch, letter_lps, logps_perm, orders, letters,
                    conf_ids, conf_vals, args, agg):
    """Joint policy gradient over (letter, confidence); proper scoring rule on confidence."""
    B, G = len(batch), args.group_size
    device = next(model.parameters()).device
    with torch.no_grad():
        samples = [torch.multinomial(lp.exp(), G, replacement=True) for lp in letter_lps]
    conf_lps, conf_samples = [], []
    for g in range(G):
        chosen = [LETTERS[int(samples[i][g])] for i in range(B)]
        clp = conf_forward(model, processor, batch, chosen, conf_ids, args)
        conf_lps.append(clp)
        with torch.no_grad():
            conf_samples.append(torch.multinomial(clp.exp(), 1).squeeze(1))
    loss = None
    for i, it in enumerate(batch):
        with torch.no_grad():
            inv = torch.argsort(torch.tensor(orders[i]))
            pi_perm_corr = logps_perm[i].exp()[inv]
            greedy = int(letter_lps[i].detach().argmax())
            y = (samples[i] == int(it["answer"])).float()
            conf_v = torch.tensor([conf_vals[int(conf_samples[g][i])] for g in range(G)])
            proper = torch.tensor([proper_score(float(conf_v[g]), float(y[g]), args.conf_reward)
                                   for g in range(G)])
            reward = y + args.lambda_consistency * pi_perm_corr[samples[i]] + args.lambda_conf * proper
            adv = (reward - reward.mean()) / (reward.std() + 1e-4)
            agg["reward"] += reward.mean().item()
            agg["correct"] += y.mean().item()
            agg["greedy"] += float(greedy == int(it["answer"]))
            agg["consistency"] += float(int(pi_perm_corr.argmax()) == greedy)
            agg["confidence"] += conf_v.mean().item()
            agg["conf_brier"] += float(((conf_v - y) ** 2).mean())
            agg["n"] += 1
        item_loss = torch.zeros((), device=device)
        for g in range(G):
            a = int(samples[i][g])
            c = int(conf_samples[g][i])
            item_loss = item_loss - adv[g].to(device) * (letter_lps[i][a].to(device)
                                                         + conf_lps[g][i][c].to(device))
        loss = item_loss if loss is None else loss + item_loss
    return loss / (B * G)


@torch.inference_mode()
def eval_confidence(model, pool, args, processor, letters, conf_ids, conf_vals):
    """Greedy letter + argmax confidence; ECE/Brier of the verbalized confidence."""
    from train_probe import ece
    model.eval()
    per = {}
    if not pool:
        return per
    for start in range(0, len(pool), args.eval_batch):
        batch = [dict(it, images=[Image.open(io.BytesIO(b)).convert("RGB") for b in it["image_bytes"]])
                 for it in pool[start:start + args.eval_batch]]
        inputs = prepare(processor, batch, args.image_size, args.pad_multiple)
        logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :].float().cpu()
        greedy = []
        for j, it in enumerate(batch):
            probs = torch.softmax(logits[j, letters[:len(it["options"])]], dim=-1)
            greedy.append(LETTERS[int(probs.argmax())])
        clp = conf_forward(model, processor, batch, greedy, conf_ids, args)
        for j, it in enumerate(batch):
            d = per.setdefault(it["subset"], {"n": 0, "correct": 0, "confs": [], "labels": []})
            correct = int(greedy[j] == LETTERS[int(it["answer"])])
            d["n"] += 1
            d["correct"] += correct
            d["confs"].append(conf_vals[int(clp[j].argmax())])
            d["labels"].append(correct)
    out = {}
    for s, d in per.items():
        confs = np.array(d["confs"], dtype=float)
        labels = np.array(d["labels"], dtype=float)
        out[s] = {"n": d["n"], "acc": d["correct"] / max(1, d["n"]), "mean_conf": float(confs.mean()),
                  "conf_ece": ece(np.clip(confs, 1e-6, 1 - 1e-6), labels)[0],
                  "conf_brier": float(((confs - labels) ** 2).mean())}
    return out


def permuted_batch(batch, step):
    """Permute options per item; order[p] = original index shown at permuted position p."""
    out, orders = [], []
    for it in batch:
        k = len(it["options"])
        rng = random.Random(f"{it['image_key']}:{step}")
        order = list(range(k))
        rng.shuffle(order)
        out.append({**it, "options": [it["options"][j] for j in order],
                    "answer": order.index(it["answer"])})
        orders.append(order)
    return out, orders


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "cauldron")
    ap.add_argument("--train-subsets", default="tallyqa:400,nlvr2:400,iconqa:300,clevr:200,vqav2:200")
    ap.add_argument("--eval-subsets", default="tallyqa:150,nlvr2:150,iconqa:150")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ap.add_argument("--init-adapter", type=Path, default=None, help="start policy (SFT/RFT adapter)")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lambda-consistency", type=float, default=0.5)
    ap.add_argument("--confidence", action="store_true",
                    help="also train a verbalized confidence with a strictly proper scoring rule")
    ap.add_argument("--lambda-conf", type=float, default=1.0)
    ap.add_argument("--conf-reward", choices=["logloss", "brier"], default="logloss")
    ap.add_argument("--beta-kl", type=float, default=0.0, help="KL to base (adapter disabled) if >0")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--targets", nargs="+", default=None)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.targets is None:
        from train_qlora import DEFAULT_TARGETS
        args.targets = DEFAULT_TARGETS

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    train_spec, eval_spec = parse_sizes(args.train_subsets), parse_sizes(args.eval_subsets)
    if args.smoke:
        train_spec, eval_spec = [(s, 24) for s, _ in train_spec[:2]], [(s, 8) for s, _ in eval_spec[:2]]
        args.group_size, args.max_steps, args.batch, args.grad_accum = 2, 3, 2, 1
        args.workers, args.eval_workers = 2, 2
        args.epochs = 1

    print("building pools ...", flush=True)
    train_pool = load_pool(args.data, train_spec, "train", args.seed)
    eval_pool = load_pool(args.data, eval_spec, "test", args.seed)
    if not train_pool:
        raise SystemExit("no training items found")

    processor = __import__("transformers").AutoProcessor.from_pretrained(
        args.model, revision=args.revision, min_pixels=128 * 32 * 32, max_pixels=448 * 32 * 32)
    processor.tokenizer.padding_side = "left"
    letters = letter_ids(processor.tokenizer)
    model = build_model(args, processor)
    conf_ids = conf_vals = None
    if args.confidence:
        conf_ids, conf_vals = confidence_ids(letters)

    history = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "before": {}, "steps": []}
    if eval_pool:
        history["before"] = eval_all(model, eval_pool, args, processor, letters)
        for s, d in history["before"].items():
            print(f"  before {s:9s} n={d['n']:4d} acc={d['acc']:.3f} ece={d['ece']:.4f}")

    optimizer = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad],
                                         lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(1, len(train_pool) // (args.batch * args.grad_accum))
    total_steps = args.max_steps or max(1, steps_per_epoch * args.epochs)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    model.train()
    step, seen, accum = 0, 0, 0
    agg = {"reward": 0.0, "correct": 0.0, "greedy": 0.0, "consistency": 0.0, "kl": 0.0,
           "confidence": 0.0, "conf_brier": 0.0, "n": 0}
    t0, done = time.perf_counter(), False
    for epoch in range(args.epochs):
        ds = ItemDataset(train_pool, shuffle_options=False, epoch=epoch)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                            collate_fn=RawCollate(), drop_last=True,
                            prefetch_factor=4 if args.workers else None)
        for batch in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logps, letter_inputs = policy_forward(model, processor, batch, letters, args)
                perm_items, orders = permuted_batch(batch, step)
                with torch.no_grad():
                    logps_perm, _ = policy_forward(model, processor, perm_items, letters, args)
                if args.beta_kl > 0:
                    with torch.no_grad(), model.disable_adapter():
                        ref_logps, _ = policy_forward(model, processor, batch, letters, args)
                if args.confidence:
                    loss = confidence_loss(model, processor, batch, logps, logps_perm, orders,
                                           letters, conf_ids, conf_vals, args, agg)
                else:
                    loss = 0.0
                for i, it in enumerate(batch):
                    if args.confidence:
                        break
                    k = len(it["options"])
                    lp = logps[i]
                    with torch.no_grad():
                        probs = lp.exp()
                        samples = torch.multinomial(probs, args.group_size, replacement=True)
                        inv = torch.argsort(torch.tensor(orders[i]))
                        pi_perm_corr = logps_perm[i].exp()[inv]
                        correct = (samples == int(it["answer"])).float()
                        reward = correct + args.lambda_consistency * pi_perm_corr[samples]
                        adv = (reward - reward.mean()) / (reward.std() + 1e-4)
                    loss = loss - (adv * lp[samples]).mean() / len(batch)
                    with torch.no_grad():
                        agg["reward"] += reward.mean().item()
                        agg["correct"] += correct.mean().item()
                        agg["greedy"] += float(int(probs.argmax()) == int(it["answer"]))
                        agg["consistency"] += float(int(pi_perm_corr.argmax()) == int(probs.argmax()))
                        if args.beta_kl > 0:
                            p = lp.exp()
                            agg["kl"] += float((p * (lp - ref_logps[i])).sum())
                        agg["n"] += 1
                if args.beta_kl > 0:
                    kl_sum = 0.0
                    for i in range(len(batch)):
                        p = logps[i].exp()
                        kl_sum = kl_sum + (p * (logps[i] - ref_logps[i])).sum()
                    loss = loss + args.beta_kl * kl_sum / len(batch)
            (loss / args.grad_accum).backward()
            accum += 1
            seen += len(batch)
            if accum % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == 1:
                    n = max(1, agg["n"])
                    print(f"step {step:5d}/{total_steps} reward {agg['reward'] / n:.3f} "
                          f"sample_acc {agg['correct'] / n:.3f} greedy {agg['greedy'] / n:.3f} "
                          f"consistency {agg['consistency'] / n:.3f} conf {agg['confidence'] / n:.3f} "
                          f"conf_brier {agg['conf_brier'] / n:.3f} kl {agg['kl'] / n:.4f} "
                          f"{(time.perf_counter() - t0) / n:.2f}s/item", flush=True)
                    agg = {k: 0.0 for k in agg}
                    agg["n"] = 0
                    t0 = time.perf_counter()
                if step >= total_steps:
                    done = True
                    break
        if done:
            break

    print("eval after ...", flush=True)
    after = eval_all(model, eval_pool, args, processor, letters)
    for s, d in after.items():
        b = history["before"].get(s, {})
        print(f"  after  {s:9s} n={d['n']:4d} acc={d['acc']:.3f} (before {b.get('acc', float('nan')):.3f}) "
              f"ece={d['ece']:.4f}")
    model.save_pretrained(args.out / "adapter")
    processor.save_pretrained(args.out / "processor")
    if args.confidence:
        conf_eval = eval_confidence(model, eval_pool, args, processor, letters, conf_ids, conf_vals)
        for s, d in conf_eval.items():
            print(f"  conf   {s:9s} n={d['n']:4d} acc={d['acc']:.3f} conf={d['mean_conf']:.3f} "
                  f"conf_ece={d['conf_ece']:.4f} conf_brier={d['conf_brier']:.4f}")
        history["after_conf"] = conf_eval
    (args.out / "metrics.json").write_text(json.dumps({"steps": step, "before": history["before"],
                                                       "after": after, "after_conf": history.get("after_conf", {}),
                                                       **history["args"]}, indent=2, default=str))
    print("saved ->", args.out / "adapter")


if __name__ == "__main__":
    main()
