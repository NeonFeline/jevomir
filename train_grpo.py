"""Policy stage for the one-pass letter reader (single-token policy, no decoding).

Policy: the softmax pi over the k option letters at the final prompt position -- identical to
inference. That softmax IS the reported probability, so the objective must reward calibration.

This file used to run GRPO (sample G letters, reward 1[a == answer] + lambda * pi_perm(a),
group-normalized advantage). That was wrong for this policy:
  * expected 0/1 reward pi(answer) is not a proper scoring rule: it is maximized by a one-hot
    pi, so the letter ECE got worse as accuracy rose;
  * the consistency bonus sum_a pi(a) pi_perm(a) ~ sum pi^2 once the model is order-invariant,
    i.e. a pure sharpening reward, right or wrong;
  * with k <= 4 actions the expectation is exact in one forward pass, so sampling only added
    noise, and the std normalization zeroed all-correct groups and blew up lucky ones.

Now every term is exact over the k letters (no sampling):
    loss = -log pi(answer)                                   (log score: proper)
         + lambda_consistency * KL(pi_perm || pi)            (order consistency, minimized at
                                                              pi == pi_perm, not at one-hot)
         + beta_kl * KL(pi || pi_ref)                        (anchor to the starting policy)
With --confidence, the verbalized confidence is trained for every letter a, weighted by the
(detached) pi(a); its proper score is NOT fed back into the letter objective (that paid the
letter policy for wrong answers the confidence head flagged).

LoRA dropout is disabled: the KL reference is computed in eval mode, and dropout in the training
forward made KL(pi || pi_ref) non-zero at step 0.

Data-parallel: torchrun --nproc_per_node 8 train_grpo.py ... ; LoRA gradients are all-reduced
before every optimizer step.
"""

import argparse
import io
import json
import math
import os
import random
import time

import numpy as np
import torch
from PIL import Image

import bitsandbytes as bnb
import dist_utils as du
from cauldron_tasks import LETTERS
from extract_cauldron import letterbox, prepare, prompt_text
from train_qlora import (ItemDataset, add_common_args, batches_per_rank, build_model, eval_all,
                         item_seed, jsonable_args, load_pool, load_processor, load_train_pool, lr_schedule,
                         make_loader, move, optimizer_step, parse_sizes, prepare_out, save_adapter,
                         seed_everything, seed_rank, trainable)
from train_probe import ece


def permuted_items(batch, salt):
    """Permute options per item; order[p] = original index shown at permuted position p.
    Never the identity (for k=2 this is always the swap)."""
    out, orders = [], []
    for it in batch:
        k = len(it["options"])
        rng = random.Random(item_seed(it, salt))
        order = list(range(k))
        while k > 1 and order == sorted(order):
            rng.shuffle(order)
        out.append({**it, "options": [it["options"][j] for j in order],
                    "answer": order.index(it["answer"])})
        orders.append(order)
    return out, orders


class GRPOCollate:
    """Worker-side: original + permuted prompts are tokenized off the training loop."""

    def __init__(self, processor, image_size, pad_multiple, epoch):
        self.processor, self.image_size, self.pad_multiple, self.epoch = processor, image_size, pad_multiple, epoch

    def __call__(self, batch):
        perm, orders = permuted_items(batch, f"perm{self.epoch}")
        return (batch, dict(prepare(self.processor, batch, self.image_size, self.pad_multiple)),
                dict(prepare(self.processor, perm, self.image_size, self.pad_multiple)), orders)


def letter_logps(model, inputs, batch, letters):
    """Per-item log-softmax over its k option letters (on GPU, differentiable if grad is on)."""
    logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :].float()
    return [torch.log_softmax(logits[i, letters[:len(it["options"])]], dim=-1) for i, it in enumerate(batch)]


CONF_TEXT = ("\nConfidence that the answer is correct, as a digit "
             "(0=0-10%, 1=10-20%, 2=20-30%, 3=30-40%, 4=40-50%, 5=50-60%, 6=60-70%, 7=70-80%, "
             "8=80-90%, 9=90-100%):")
CONF_VALS = [0.05 + 0.1 * i for i in range(10)]


def confidence_ids(tokenizer):
    """Digit tokens 0-9 as confidence buckets (centers 5%..95%). Not the A-K letters: those are
    also the answer tokens, so the answer-letter prior leaked into the confidence scale."""
    ids = []
    for d in "0123456789":
        encoded = tokenizer.encode(d, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != d:
            raise ValueError(f"Digit {d!r} is not a single token")
        ids.append(encoded[0])
    return ids, list(CONF_VALS)


def bucket_target(p, vals=CONF_VALS):
    """Two-bucket soft target whose expected value is p (clamped to the outer bucket centers)."""
    n = len(vals)
    pos = (min(max(float(p), vals[0]), vals[-1]) - vals[0]) / (vals[1] - vals[0])
    lo = min(int(pos), n - 2)
    t = torch.zeros(n)
    t[lo], t[lo + 1] = 1 - (pos - lo), pos - lo
    return t


def conf_forward(model, processor, batch, chosen_letters, conf_ids, args):
    """One pass per item: prompt + 'Answer: <letter>' + confidence scale -> log-probs over buckets."""
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
    logits = model(**move(dict(inputs), "cuda"), logits_to_keep=1).logits[:, -1, :].float()
    return torch.log_softmax(logits[:, conf_ids], dim=-1)


def proper_score(c, y, kind):
    """Strictly proper reward for a stated confidence c and correctness y in {0,1}."""
    if kind == "brier":
        return 1.0 - (c - y) ** 2
    c = min(max(c, 1e-3), 1 - 1e-3)
    return (y * math.log(c) + (1 - y) * math.log(1 - c) + math.log(2)) / math.log(2)


def pi_perm_original(logp_perm, order):
    """Permuted-order probabilities re-indexed to original option positions."""
    inv = torch.argsort(torch.tensor(order, device=logp_perm.device))
    return logp_perm.exp()[inv]


def letter_loss(batch, logps, logps_perm, orders, args, agg):
    """Exact log score on the answer + KL(pi_perm || pi) order consistency, averaged over items.
    pi_perm is a detached target: pulling pi toward it has no preference for sharp pi."""
    loss = 0.0
    for i, it in enumerate(batch):
        lp, ans = logps[i], int(it["answer"])
        with torch.no_grad():
            pi_perm = pi_perm_original(logps_perm[i], orders[i]).clamp_min(1e-8)
        cons = (pi_perm * (pi_perm.log() - lp)).sum()
        loss = loss + (-lp[ans] + args.lambda_consistency * cons) / len(batch)
        lpd = lp.detach()
        greedy = int(lpd.argmax())
        agg["nll"] += float(-lpd[ans])
        agg["p_answer"] += float(lpd[ans].exp())
        agg["greedy"] += float(greedy == ans)
        agg["consistency"] += float(int(pi_perm.argmax()) == greedy)
        agg["cons_kl"] += float(cons.detach())
        agg["n"] += 1
    return loss


def confidence_loss(model, processor, batch, letter_lps, logps_perm, orders, conf_ids, conf_vals, args, agg):
    """letter_loss + exact expected proper score on the stated confidence
    + distillation of the model's own letter probability p(letter) into the stated confidence.

    The proper score alone found a calibrated CONSTANT (overnight run: "95%" on every item),
    since nothing tells the confidence prompt which items are hard. The letter distribution
    already knows (letter ECE ~0.05), so --lambda-distill pulls the bucket distribution
    toward a target with expected value p(letter), per item.

    Exact over both letters and buckets: for every option letter a the objective is
    E_c[S(c, 1[a == answer])], weighted by the detached pi(a) (the letters the model would
    answer with). Sampling c (the first version) collapsed once p(c) was nearly one-hot, and
    the confidence score is kept out of the letter objective, so the letter policy cannot
    gain by choosing wrong answers that the confidence head flags.
    """
    B = len(batch)
    loss = letter_loss(batch, letter_lps, logps_perm, orders, args, agg)
    pairs = [(i, a) for i, it in enumerate(batch) for a in range(len(it["options"]))]
    vals = torch.tensor(conf_vals, device=letter_lps[0].device)
    chunk = max(1, 2 * B)
    for start in range(0, len(pairs), chunk):
        part = pairs[start:start + chunk]
        clp = conf_forward(model, processor, [batch[i] for i, _ in part],
                           [LETTERS[a] for _, a in part], conf_ids, args)
        for row, (i, a) in enumerate(part):
            y = float(a == int(batch[i]["answer"]))
            w = float(letter_lps[i][a].detach().exp())
            p = clp[row].exp()
            scores = torch.tensor([proper_score(float(v), y, args.conf_reward) for v in conf_vals],
                                  device=p.device)
            # Trained on a temperature-softened q = softmax(logits / T): same best bucket, but the
            # gradient q_c (S_c - E_q S) does not vanish when p is already near one-hot.
            q = torch.softmax(clp[row] / args.conf_temperature, dim=-1)
            loss = loss - args.lambda_conf * w * (q * scores).sum() / B
            if args.lambda_distill:
                target = bucket_target(w).to(p.device)
                loss = loss - args.lambda_distill * w * (target * clp[row]).sum() / B
            with torch.no_grad():
                c = float((p * vals).sum())
                agg["conf_score"] += w * float((p * scores).sum())
                agg["confidence"] += w * c
                agg["conf_sq"] += w * c * c
                agg["conf_brier"] += w * float((p * (vals - y) ** 2).sum())
    return loss


@torch.inference_mode()
def eval_confidence(model, pool, args, processor, letters, conf_ids, conf_vals):
    """Greedy letter + argmax confidence; ECE/Brier of the verbalized confidence (sharded)."""
    model.eval()
    per = {}
    mine = du.shard(pool)
    for start in range(0, len(mine), args.eval_batch):
        batch = [dict(it, images=[Image.open(io.BytesIO(b)).convert("RGB") for b in it["image_bytes"]])
                 for it in mine[start:start + args.eval_batch]]
        inputs = dict(prepare(processor, batch, args.image_size, args.pad_multiple))
        logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :].float().cpu()
        greedy = [LETTERS[int(logits[j, letters[:len(it["options"])]].argmax())] for j, it in enumerate(batch)]
        clp = conf_forward(model, processor, batch, greedy, conf_ids, args).cpu()
        for j, it in enumerate(batch):
            d = per.setdefault(it["subset"], {"n": 0, "correct": 0, "confs": [], "labels": []})
            correct = int(greedy[j] == LETTERS[int(it["answer"])])
            d["n"] += 1
            d["correct"] += correct
            d["confs"].append(conf_vals[int(clp[j].argmax())])
            d["labels"].append(correct)
    model.train()
    merged = {}
    for part in du.gather(per):
        for s, d in part.items():
            m = merged.setdefault(s, {"n": 0, "correct": 0, "confs": [], "labels": []})
            for k in m:
                m[k] += d[k]
    out = {}
    for s, d in sorted(merged.items()):
        confs = np.array(d["confs"], dtype=float)
        labels = np.array(d["labels"], dtype=float)
        out[s] = {"n": d["n"], "acc": d["correct"] / max(1, d["n"]), "mean_conf": float(confs.mean()),
                  "conf_ece": ece(np.clip(confs, 1e-6, 1 - 1e-6), labels)[0],
                  "conf_brier": float(((confs - labels) ** 2).mean()),
                  "conf_sd": float(confs.std()), "conf_auroc": auroc(confs, labels)}
    return out


def auroc(scores, labels):
    """P(score of a correct item > score of a wrong one), ties count half; 0.5 = no discrimination."""
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


class RefCollate:
    def __init__(self, processor, image_size, pad_multiple):
        self.processor, self.image_size, self.pad_multiple = processor, image_size, pad_multiple

    def __call__(self, batch):
        return batch, dict(prepare(self.processor, batch, self.image_size, self.pad_multiple))


@torch.inference_mode()
def reference_logps(model, pool, args, processor, letters):
    """Letter log-probs of the starting (SFT/RFT) policy for every train item, same prompts as
    training (this stage never shuffles options). The KL anchor then keeps the answer policy near
    the stage it started from, not near the base model the old --beta-kl pulled toward."""
    from torch.utils.data import DataLoader
    model.eval()
    mine = du.shard(list(range(len(pool))))
    out = {}
    if mine:
        loader = DataLoader(ItemDataset([pool[i] for i in mine]), batch_size=args.eval_batch,
                            num_workers=args.eval_workers,
                            collate_fn=RefCollate(processor, args.image_size, args.pad_multiple))
        for batch, inputs in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lps = letter_logps(model, inputs, batch, letters)
            for it, lp in zip(batch, lps):
                out[mine[it["index"]]] = lp.float().cpu()
    model.train()
    merged = {}
    for part in du.gather(out):
        merged.update(part)
    return merged


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.set_defaults(eval_subsets="tallyqa:150,nlvr2:150,iconqa:150", grad_accum=2, lr=1e-5,
                    warmup=10, workers=4)
    ap.add_argument("--lambda-consistency", type=float, default=0.5,
                    help="weight of KL(pi_perm || pi), the option-order consistency term")
    ap.add_argument("--confidence", action="store_true",
                    help="also train a verbalized confidence with a strictly proper scoring rule")
    ap.add_argument("--lambda-conf", type=float, default=1.0)
    ap.add_argument("--conf-reward", choices=["logloss", "brier"], default="logloss")
    ap.add_argument("--conf-temperature", type=float, default=4.0,
                    help="softmax temperature of the confidence-bucket distribution in the training objective")
    ap.add_argument("--lambda-distill", type=float, default=0.0,
                    help="pull the stated confidence toward the model's own letter probability")
    ap.add_argument("--beta-kl", type=float, default=0.0, help="KL of the letter policy to --kl-ref if >0")
    ap.add_argument("--kl-ref", choices=["init", "base"], default="init",
                    help="KL reference: the starting adapter's policy (default) or the base model")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()

    du.init()
    try:
        run(args)
    finally:
        du.cleanup()


def run(args):
    seed_everything(args.seed)
    prepare_out(args.out)
    train_spec, eval_spec = parse_sizes(args.train_subsets), parse_sizes(args.eval_subsets)
    if args.smoke:
        n = 12 * du.world_size()
        train_spec, eval_spec = [(s, n) for s, _ in train_spec[:2]], [(s, n) for s, _ in eval_spec[:2]]
        args.max_steps, args.batch, args.grad_accum = 3, 2, 1
        args.workers, args.eval_workers = 2, 2
        args.epochs = 1

    du.print0(f"world size {du.world_size()}; building pools ...")
    train_pool = load_train_pool(args, train_spec)
    eval_pool = load_pool(args.data, eval_spec, "test", args.seed)
    if not train_pool:
        raise SystemExit("no training items found")

    processor, letters = load_processor(args)
    model = build_model(args)
    # No dropout: train-mode logps must match the eval-mode KL reference (and the policy
    # that is evaluated), else KL(pi || pi_ref) is non-zero before the first update.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    seed_rank(args.seed)
    conf_ids = conf_vals = None
    if args.confidence:
        conf_ids, conf_vals = confidence_ids(processor.tokenizer)

    history = {"before": {}}
    if eval_pool:
        history["before"] = eval_all(model, eval_pool, args, processor, letters)
        for s, d in history["before"].items():
            du.print0(f"  before {s:9s} n={d['n']:4d} acc={d['acc']:.3f} ece={d['ece']:.4f}")
        if args.confidence:
            history["before_conf"] = eval_confidence(model, eval_pool, args, processor, letters,
                                                     conf_ids, conf_vals)

    ref = None
    if args.beta_kl > 0 and args.kl_ref == "init":
        du.print0(f"reference letter log-probs for {len(train_pool)} train items ...")
        ref = reference_logps(model, train_pool, args, processor, letters)

    steps_per_epoch = batches_per_rank(len(train_pool), args.batch) // args.grad_accum
    if steps_per_epoch == 0:
        raise SystemExit(f"{len(train_pool)} items < one global batch "
                         f"({args.batch} x {args.grad_accum} x {du.world_size()} ranks)")
    total_steps = min(args.max_steps, steps_per_epoch * args.epochs) if args.max_steps \
        else steps_per_epoch * args.epochs
    lr_at, warmup = lr_schedule(args.lr, args.warmup, total_steps)
    du.print0(f"policy stage: {len(train_pool)} items, global batch {args.batch * args.grad_accum * du.world_size()}, "
              f"{total_steps} steps, warmup {warmup}")
    optimizer = bnb.optim.PagedAdamW8bit(trainable(model), lr=args.lr, weight_decay=0.0)

    model.train()
    step, micro, skipped, last_log = 0, 0, 0, 0
    fresh = lambda: {"nll": 0.0, "p_answer": 0.0, "greedy": 0.0, "consistency": 0.0, "cons_kl": 0.0,
                     "kl": 0.0, "confidence": 0.0, "conf_brier": 0.0, "conf_sq": 0.0, "conf_score": 0.0,
                     "n": 0.0}
    agg = fresh()
    t0, done = time.perf_counter(), False
    deadline = t0 + args.time_budget_min * 60 if args.time_budget_min else None
    for epoch in range(args.epochs):
        loader, sampler = make_loader(ItemDataset(train_pool), args.batch, args.workers,
                                      GRPOCollate(processor, args.image_size, args.pad_multiple, epoch),
                                      args.seed, shuffle=True, drop_last=True)
        sampler.set_epoch(epoch)
        usable = (len(loader) // args.grad_accum) * args.grad_accum
        for b_idx, (batch, inputs, perm_inputs, orders) in enumerate(loader):
            if b_idx >= usable:
                break
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logps = letter_logps(model, inputs, batch, letters)
                with torch.no_grad():
                    logps_perm = letter_logps(model, perm_inputs, batch, letters)
                if args.confidence:
                    loss = confidence_loss(model, processor, batch, logps, logps_perm, orders,
                                           conf_ids, conf_vals, args, agg)
                else:
                    loss = letter_loss(batch, logps, logps_perm, orders, args, agg)
                if args.beta_kl > 0:
                    if ref is not None:
                        ref_logps = [ref[int(it["index"])].to(logps[0].device) for it in batch]
                    else:
                        with torch.no_grad(), model.disable_adapter():
                            ref_logps = letter_logps(model, inputs, batch, letters)
                    kl = sum((lp.exp() * (lp - ref)).sum() for lp, ref in zip(logps, ref_logps)) / len(batch)
                    loss = loss + args.beta_kl * kl
                    agg["kl"] += float(kl.detach()) * len(batch)
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            norm, ok = optimizer_step(model, optimizer)
            skipped += int(not ok)
            step += 1
            if step % args.log_every == 0 or step == 1 or step == total_steps:
                tot = du.all_reduce_sum(agg)
                n = max(1.0, tot["n"])
                du.print0(f"step {step:5d}/{total_steps} nll {tot['nll'] / n:.3f} "
                          f"p_answer {tot['p_answer'] / n:.3f} greedy {tot['greedy'] / n:.3f} "
                          f"consistency {tot['consistency'] / n:.3f} cons_kl {tot['cons_kl'] / n:.4f} "
                          f"conf {tot['confidence'] / n:.3f} conf_score {tot['conf_score'] / n:.3f} "
                          f"conf_sd {math.sqrt(max(0.0, tot['conf_sq'] / n - (tot['confidence'] / n) ** 2)):.3f} "
                          f"conf_brier {tot['conf_brier'] / n:.3f} kl {tot['kl'] / n:.4f} gnorm {norm:.3f} "
                          f"{(time.perf_counter() - t0) / (step - last_log):.2f}s/step"
                          + ("" if ok else " NONFINITE-SKIPPED"))
                agg, t0, last_log = fresh(), time.perf_counter(), step
            if args.save_every and step % args.save_every == 0 and step < total_steps:
                save_adapter(model, args.out / f"checkpoint-{step}")
            if step >= total_steps:
                done = True
                break
            if deadline and du.any_true(time.perf_counter() > deadline):
                du.print0(f"time budget reached at step {step}/{total_steps}")
                done = True
                break
        if done:
            break

    du.print0("eval after ...")
    after = eval_all(model, eval_pool, args, processor, letters)
    for s, d in after.items():
        b = history["before"].get(s, {})
        du.print0(f"  after  {s:9s} n={d['n']:4d} acc={d['acc']:.3f} (before {b.get('acc', float('nan')):.3f}) "
                  f"ece={d['ece']:.4f}")
    if args.confidence and eval_pool:
        conf_eval = eval_confidence(model, eval_pool, args, processor, letters, conf_ids, conf_vals)
        for s, d in conf_eval.items():
            du.print0(f"  conf   {s:9s} n={d['n']:4d} acc={d['acc']:.3f} conf={d['mean_conf']:.3f} "
                      f"conf_ece={d['conf_ece']:.4f} conf_brier={d['conf_brier']:.4f} "
                      f"conf_sd={d['conf_sd']:.3f} conf_auroc={d['conf_auroc']:.3f}")
        history["after_conf"] = conf_eval
    save_adapter(model, args.out / "adapter")
    if du.is_main():
        processor.save_pretrained(args.out / "processor")
        (args.out / "metrics.json").write_text(json.dumps({
            "steps": step, "planned_steps": total_steps, "skipped_nonfinite": skipped, "world_size": du.world_size(),
            "global_batch": args.batch * args.grad_accum * du.world_size(),
            "before": history["before"], "after": after,
            "before_conf": history.get("before_conf", {}), "after_conf": history.get("after_conf", {}),
            "args": jsonable_args(args)}, indent=2, default=str))
    du.print0("saved ->", args.out / "adapter")


if __name__ == "__main__":
    main()
