"""QLoRA SFT for the one-pass letter reader.

Training objective == inference objective: one forward pass over the prompt
(image + question + options + instruction + assistant scaffold). The loss is
computed ONLY on the final position, whose target is the answer-letter token.
No prompt token contributes to the loss, so the model cannot "learn the prompt".

Options are optionally shuffled per epoch (the answer index follows) to reduce
the position bias noted in API.md. Evaluates accuracy / raw ECE per subset
before and after training.
"""

import argparse
import hashlib
import io
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from cauldron_tasks import build_index, image_key
from extract_cauldron import letter_ids, prepare
from train_probe import ece
from train_probe_cauldron import split_of

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

DEFAULT_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",                           # 8 full-attention layers
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",  # 24 gated-delta layers
    "gate_proj", "up_proj", "down_proj",                               # MLP
]


def parse_sizes(spec):
    out = []
    for part in spec.split(","):
        name, _, limit = part.partition(":")
        out.append((name.strip(), int(limit)))
    return out


def load_pool(data_dir, subsets, want, seed, test_frac=0.15, val_frac=0.1):
    """Items whose perceptual image hash falls in the requested split bucket."""
    pool = []
    for subset, need in subsets:
        got = 0
        for it in build_index(subset, Path(data_dir), need * 3 + 50, seed=seed):
            key = image_key([Image.open(io.BytesIO(b)).convert("RGB") for b in it["image_bytes"]])
            if split_of(key, test_frac, val_frac) != want:
                continue
            pool.append({
                "subset": subset, "image_id": it["image_id"], "image_key": key,
                "question": it["question"], "options": list(it["options"]),
                "answer": it["answer"], "answer_kind": it["answer_kind"],
                "image_bytes": it["image_bytes"],
            })
            got += 1
            if got >= need:
                break
        print(f"  pool[{want}] {subset}: {got} items", flush=True)
    return pool


class ItemDataset(Dataset):
    def __init__(self, items, shuffle_options=False, epoch=0):
        self.items = items
        self.shuffle_options = shuffle_options
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        options, answer = list(it["options"]), it["answer"]
        if self.shuffle_options:
            key = int(hashlib.sha256(str(it["image_id"]).encode()).hexdigest()[:8], 16)
            rng = random.Random(key ^ (self.epoch * 1_000_003))
            order = list(range(len(options)))
            rng.shuffle(order)
            options = [options[j] for j in order]
            answer = order.index(answer)
        return {
            "index": i, "images": [Image.open(io.BytesIO(b)).convert("RGB") for b in it["image_bytes"]],
            "question": it["question"], "options": options, "answer": answer,
            "subset": it["subset"], "answer_kind": it["answer_kind"], "image_key": it["image_key"],
        }


class Collate:
    def __init__(self, processor, letter_token_ids, image_size, pad_multiple):
        self.processor, self.letter_token_ids = processor, letter_token_ids
        self.image_size, self.pad_multiple = image_size, pad_multiple

    def __call__(self, batch):
        inputs = prepare(self.processor, batch, self.image_size, self.pad_multiple)
        answer = torch.tensor([self.letter_token_ids[b["answer"]] for b in batch])
        meta = [{"subset": b["subset"], "answer_kind": b["answer_kind"],
                 "n_options": len(b["options"]), "index": b.get("index", -1)} for b in batch]
        return meta, dict(inputs), answer


def move(inputs, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in inputs.items()}


@torch.inference_mode()
def evaluate(model, items, processor, letter_token_ids, batch, workers, image_size, pad_multiple):
    """Per-subset accuracy, mean raw confidence and raw ECE over the option letters."""
    per_subset = {}
    if not items:
        return per_subset
    letter_index = {tid: i for i, tid in enumerate(letter_token_ids)}
    was_training = model.training
    model.eval()
    try:
        loader = DataLoader(ItemDataset(items), batch_size=batch, num_workers=workers,
                            collate_fn=Collate(processor, letter_token_ids, image_size, pad_multiple),
                            pin_memory=True)
        for meta, inputs, answer in loader:
            logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :].float().cpu()
            letters = logits[:, letter_token_ids]
            for j, m in enumerate(meta):
                probs = torch.softmax(letters[j, :m["n_options"]], dim=-1)
                correct = int(int(probs.argmax()) == letter_index[int(answer[j])])
                d = per_subset.setdefault(m["subset"], {"n": 0, "correct": 0, "confs": [], "labels": [], "kinds": {}})
                d["n"] += 1
                d["correct"] += correct
                d["confs"].append(float(probs.max()))
                d["labels"].append(correct)
                kind = d["kinds"].setdefault(m["answer_kind"], {"n": 0, "correct": 0})
                kind["n"] += 1
                kind["correct"] += correct
    finally:
        if was_training:
            model.train()
    return per_subset


def summarize(per_subset):
    out = {}
    for sub, d in per_subset.items():
        confs = np.clip(np.array(d["confs"]), 1e-6, 1 - 1e-6)
        labels = np.array(d["labels"], dtype=float)
        out[sub] = {
            "n": d["n"], "acc": d["correct"] / max(1, d["n"]), "mean_conf": float(confs.mean()),
            "ece": ece(confs, labels)[0],
            "kinds": {k: {"n": v["n"], "acc": v["correct"] / max(1, v["n"])} for k, v in d["kinds"].items()},
        }
    return out


def eval_all(model, pool, args, processor, letters):
    raw = {}
    for sub in sorted({it["subset"] for it in pool}):
        items = [it for it in pool if it["subset"] == sub]
        raw.update(evaluate(model, items, processor, letters, args.eval_batch,
                            args.eval_workers, args.image_size, args.pad_multiple))
    return summarize(raw)


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "cauldron")
    ap.add_argument("--train-subsets", default="tallyqa:400,nlvr2:400,iconqa:300,clevr:200,vqav2:200")
    ap.add_argument("--eval-subsets", default="tallyqa:150,nlvr2:150,iconqa:150,clevr:100,vqav2:100")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="overrides epochs when > 0")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--shuffle-options", action="store_true")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    train_spec, eval_spec = parse_sizes(args.train_subsets), parse_sizes(args.eval_subsets)
    if args.smoke:
        train_spec, eval_spec = [(s, 8) for s, _ in train_spec[:2]], [(s, 8) for s, _ in eval_spec[:2]]
        args.max_steps = args.max_steps or 5
        args.batch, args.grad_accum, args.workers, args.eval_workers = 2, 1, 2, 2
        args.epochs = 1

    print("building pools ...", flush=True)
    train_pool = load_pool(args.data, train_spec, "train", args.seed)
    eval_pool = load_pool(args.data, eval_spec, "test", args.seed)
    if not train_pool:
        raise SystemExit("no training items found; check --data and --train-subsets")

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                              min_pixels=128 * 32 * 32, max_pixels=448 * 32 * 32)
    processor.tokenizer.padding_side = "left"
    letters = letter_ids(processor.tokenizer)

    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, quantization_config=quant,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", target_modules=args.targets))
    model.print_trainable_parameters()

    before = {}
    if eval_pool:
        print("eval before ...", flush=True)
        before = eval_all(model, eval_pool, args, processor, letters)
        for s, d in before.items():
            print(f"  before {s:9s} n={d['n']:4d} acc={d['acc']:.3f} conf={d['mean_conf']:.3f} ece={d['ece']:.4f}")

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
    step, seen, log_loss, log_n = 0, 0, 0.0, 0
    t0, done = time.perf_counter(), False
    for epoch in range(args.epochs):
        dataset = ItemDataset(train_pool, shuffle_options=args.shuffle_options, epoch=epoch)
        loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                            collate_fn=Collate(processor, letters, args.image_size, args.pad_multiple),
                            pin_memory=True, drop_last=True,
                            prefetch_factor=4 if args.workers else None)
        for meta, inputs, answer in loader:
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :]
                loss = F.cross_entropy(logits.float(), answer.cuda())
            (loss / args.grad_accum).backward()
            log_loss += loss.item()
            log_n += 1
            seen += len(answer)
            if log_n % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0 or step == 1:
                    print(f"step {step:5d}/{total_steps} loss {log_loss / log_n:.4f} lr {lr_at(step):.2e} "
                          f"seen {seen} {(time.perf_counter() - t0) / max(1, log_n):.2f}s/batch", flush=True)
                    t0, log_loss, log_n = time.perf_counter(), 0.0, 0
                if step >= total_steps:
                    done = True
                    break
        if done:
            break

    print("eval after ...", flush=True)
    after = eval_all(model, eval_pool, args, processor, letters)
    for s, d in after.items():
        b = before.get(s, {})
        print(f"  after  {s:9s} n={d['n']:4d} acc={d['acc']:.3f} (before {b.get('acc', float('nan')):.3f}) "
              f"conf={d['mean_conf']:.3f} ece={d['ece']:.4f}")

    model.save_pretrained(args.out / "adapter")
    processor.save_pretrained(args.out / "processor")
    (args.out / "metrics.json").write_text(json.dumps({
        "train_items": len(train_pool), "eval_items": len(eval_pool), "steps": step,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "targets": args.targets, "4bit": not args.no_4bit},
        "shuffle_options": args.shuffle_options, "before": before, "after": after,
    }, indent=2))
    print("saved adapter ->", args.out / "adapter")


if __name__ == "__main__":
    main()
