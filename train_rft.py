"""RFT / rejection-sampling fine-tuning for the one-pass letter reader.

Round: sample K answer letters per item from the current policy's option distribution
(one forward pass, no decoding), keep items with at least one correct sample (optionally
weighted by the number of correct samples), then SFT on them with the train_qlora
objective: loss ONLY on the answer-letter token at the final position.

Repeat for `--rounds`, evaluating accuracy / raw confidence / ECE per subset each round.
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import bitsandbytes as bnb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from extract_cauldron import letter_ids
from train_qlora import (DEFAULT_TARGETS, MODEL_ID, MODEL_REVISION, Collate, ItemDataset,
                         eval_all, load_pool, move, parse_sizes)


@torch.inference_mode()
def rollout(model, items, processor, letters, args):
    """Sample K letters per item from the option distribution; keep items with >=1 correct."""
    model.eval()
    loader = DataLoader(ItemDataset(items), batch_size=args.rollout_batch, num_workers=args.workers,
                        collate_fn=Collate(processor, letters, args.image_size, args.pad_multiple),
                        pin_memory=True)
    kept, stats = [], {}
    letter_index = {tid: i for i, tid in enumerate(letters)}
    for meta, inputs, answer in loader:
        logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :].float().cpu()
        for j, m in enumerate(meta):
            k, idx = m["n_options"], m["index"]
            probs = torch.softmax(logits[j, letters[:k]] / args.temperature, dim=-1)
            truth = letter_index[int(answer[j])]
            top1 = int(probs.argmax())
            samples = torch.multinomial(probs, args.k, replacement=True)
            n_correct = int((samples == truth).sum())
            s = stats.setdefault(m["subset"], {"n": 0, "top1": 0, "pass": 0, "samples": 0, "p_true": 0.0})
            s["n"] += 1
            s["top1"] += int(top1 == truth)
            s["pass"] += int(n_correct > 0)
            s["samples"] += n_correct
            s["p_true"] += float(probs[truth])
            if n_correct > 0:
                kept.append((items[idx], n_correct))
    for s in stats.values():
        s["top1_acc"] = s["top1"] / max(1, s["n"])
        s["pass@K"] = s["pass"] / max(1, s["n"])
        s["mean_p_true"] = s["p_true"] / max(1, s["n"])
    model.train()
    return kept, stats


def sft_round(model, items, processor, letters, args, round_idx):
    optimizer = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad],
                                         lr=args.lr, weight_decay=0.0)
    total_steps = args.sft_max_steps or max(1, len(items) // (args.batch * args.grad_accum) * args.sft_epochs)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        p = (step - args.warmup) / max(1, total_steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    model.train()
    step, log_loss, log_n, seen = 0, 0.0, 0, 0
    t0, done = time.perf_counter(), False
    for epoch in range(args.sft_epochs):
        ds = ItemDataset(items, shuffle_options=args.shuffle_options,
                         epoch=round_idx * args.sft_epochs + epoch)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
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
                    print(f"  round {round_idx + 1} step {step:4d}/{total_steps} loss {log_loss / log_n:.4f} "
                          f"lr {lr_at(step):.2e} seen {seen} {(time.perf_counter() - t0) / max(1, log_n):.2f}s/batch",
                          flush=True)
                    t0, log_loss, log_n = time.perf_counter(), 0.0, 0
                if step >= total_steps:
                    done = True
                    break
        if done:
            break
    return {"steps": step, "items": len(items), "epochs": args.sft_epochs}


def build_model(args, processor):
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
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=args.targets))
    model.print_trainable_parameters()
    return model


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "cauldron")
    ap.add_argument("--train-subsets", default="tallyqa:400,nlvr2:400,iconqa:300,clevr:200,vqav2:200")
    ap.add_argument("--eval-subsets", default="tallyqa:150,nlvr2:150,iconqa:150,clevr:100,vqav2:100")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--init-adapter", type=Path, default=None)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--k", type=int, default=8, help="samples per item per round")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--weight-by-correct", action="store_true")
    ap.add_argument("--sft-epochs", type=int, default=1)
    ap.add_argument("--sft-max-steps", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--rollout-batch", type=int, default=8)
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
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    train_spec, eval_spec = parse_sizes(args.train_subsets), parse_sizes(args.eval_subsets)
    if args.smoke:
        train_spec, eval_spec = [(s, 24) for s, _ in train_spec[:2]], [(s, 8) for s, _ in eval_spec[:2]]
        args.rounds, args.k, args.sft_max_steps = 1, 2, 5
        args.batch, args.rollout_batch, args.grad_accum, args.workers, args.eval_workers = 2, 2, 1, 2, 2
        args.sft_epochs = 1

    print("building pools ...", flush=True)
    train_pool = load_pool(args.data, train_spec, "train", args.seed)
    eval_pool = load_pool(args.data, eval_spec, "test", args.seed)
    if not train_pool:
        raise SystemExit("no training items found")

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                              min_pixels=128 * 32 * 32, max_pixels=448 * 32 * 32)
    processor.tokenizer.padding_side = "left"
    letters = letter_ids(processor.tokenizer)
    model = build_model(args, processor)

    history = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
               "before": {}, "rounds": []}
    if eval_pool:
        print("eval before ...", flush=True)
        history["before"] = eval_all(model, eval_pool, args, processor, letters)
        for s, d in history["before"].items():
            print(f"  before {s:9s} n={d['n']:4d} acc={d['acc']:.3f} ece={d['ece']:.4f}")

    for r in range(args.rounds):
        print(f"round {r + 1}/{args.rounds}: rollout K={args.k} ...", flush=True)
        kept, stats = rollout(model, train_pool, processor, letters, args)
        for s, d in sorted(stats.items()):
            print(f"  rollout {s:9s} n={d['n']:4d} top1={d['top1_acc']:.3f} pass@{args.k}={d['pass@K']:.3f} "
                  f"p_true={d['mean_p_true']:.3f}", flush=True)
        sft_items = [it for it, _ in kept]
        if args.weight_by_correct:
            sft_items = [it for it, nc in kept for _ in range(min(nc, args.k))]
        print(f"  kept {len(kept)}/{len(train_pool)} items, SFT on {len(sft_items)} ...", flush=True)
        sft = sft_round(model, sft_items, processor, letters, args, r)
        ev = eval_all(model, eval_pool, args, processor, letters) if eval_pool else {}
        for s, d in ev.items():
            b = history["before"].get(s, {})
            print(f"  after  {s:9s} n={d['n']:4d} acc={d['acc']:.3f} "
                  f"(before {b.get('acc', float('nan')):.3f}) ece={d['ece']:.4f}", flush=True)
        history["rounds"].append({"round": r + 1, "kept": len(kept), "sft": sft,
                                  "rollout": stats, "eval": ev})
        model.save_pretrained(args.out / f"adapter-round{r + 1}")

    model.save_pretrained(args.out / "adapter")
    processor.save_pretrained(args.out / "processor")
    (args.out / "metrics.json").write_text(json.dumps(history, indent=2))
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
