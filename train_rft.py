"""RFT / rejection-sampling fine-tuning for the one-pass letter reader.

Round: sample K answer letters per item from the current policy's option distribution
(one forward pass, no decoding), keep items with at least one correct sample (optionally
weighted by the number of correct samples), then SFT on them with the train_qlora
objective: loss ONLY on the answer-letter token at the final position.

Repeat for `--rounds`, evaluating accuracy / raw confidence / ECE per subset each round.
Rollouts are sharded over ranks and the kept indices gathered, so every rank then trains
on the identical kept list (torchrun --nproc_per_node 8 train_rft.py ...).
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

import dist_utils as du
from train_qlora import (Collate, ItemDataset, add_common_args, build_model, eval_all,
                         jsonable_args, load_pool, load_processor, load_train_pool, move, parse_sizes, prepare_out,
                         save_adapter, seed_everything, seed_rank, sft_train)


@torch.inference_mode()
def rollout(model, items, processor, letters, args):
    """Sample K letters per item from the option distribution; keep items with >=1 correct.

    Returns [(global index, n_correct)] and per-subset stats, identical on every rank."""
    model.eval()
    idx_all = list(range(len(items)))
    mine = du.shard(idx_all)
    kept, stats = [], {}
    letter_index = {tid: i for i, tid in enumerate(letters)}
    if mine:
        loader = DataLoader(ItemDataset([items[i] for i in mine]), batch_size=args.rollout_batch,
                            num_workers=args.workers,
                            collate_fn=Collate(processor, letters, args.image_size, args.pad_multiple),
                            pin_memory=True)
        for meta, inputs, answer in loader:
            logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :].float().cpu()
            for j, m in enumerate(meta):
                k, idx = m["n_options"], mine[m["index"]]
                probs = torch.softmax(logits[j, letters[:k]] / args.temperature, dim=-1)
                truth = letter_index[int(answer[j])]
                samples = torch.multinomial(probs, args.k, replacement=True)
                n_correct = int((samples == truth).sum())
                s = stats.setdefault(m["subset"], {"n": 0, "top1": 0, "pass": 0, "samples": 0, "p_true": 0.0})
                s["n"] += 1
                s["top1"] += int(int(probs.argmax()) == truth)
                s["pass"] += int(n_correct > 0)
                s["samples"] += n_correct
                s["p_true"] += float(probs[truth])
                if n_correct > 0:
                    kept.append((idx, n_correct))
    model.train()

    all_kept, merged = [], {}
    for part_kept, part_stats in du.gather((kept, stats)):
        all_kept += part_kept
        for sub, s in part_stats.items():
            m = merged.setdefault(sub, {k: 0 for k in s})
            for k, v in s.items():
                m[k] += v
    all_kept.sort()  # rank-independent order, so every rank builds the same SFT list
    for s in merged.values():
        s["top1_acc"] = s["top1"] / max(1, s["n"])
        s["pass@K"] = s["pass"] / max(1, s["n"])
        s["mean_p_true"] = s["p_true"] / max(1, s["n"])
    return all_kept, dict(sorted(merged.items()))


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--k", type=int, default=8, help="samples per item per round")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--weight-by-correct", action="store_true")
    ap.add_argument("--sft-epochs", type=int, default=1)
    ap.add_argument("--sft-max-steps", type=int, default=0)
    ap.add_argument("--rollout-batch", type=int, default=16)
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
        args.rounds, args.k, args.sft_max_steps = 1, 2, 5
        args.batch, args.rollout_batch, args.grad_accum, args.workers, args.eval_workers = 2, 2, 1, 2, 2
        args.sft_epochs = 1

    du.print0(f"world size {du.world_size()}; building pools ...")
    train_pool = load_train_pool(args, train_spec)
    eval_pool = load_pool(args.data, eval_spec, "test", args.seed)
    if not train_pool:
        raise SystemExit("no training items found")

    processor, letters = load_processor(args)
    model = build_model(args)
    seed_rank(args.seed)  # rollout sampling differs per rank; pools/LoRA init already agree

    history = {"args": jsonable_args(args), "world_size": du.world_size(), "before": {}, "rounds": []}
    if eval_pool:
        du.print0("eval before ...")
        history["before"] = eval_all(model, eval_pool, args, processor, letters)
        for s, d in history["before"].items():
            du.print0(f"  before {s:9s} n={d['n']:4d} acc={d['acc']:.3f} ece={d['ece']:.4f}")

    for r in range(args.rounds):
        du.print0(f"round {r + 1}/{args.rounds}: rollout K={args.k} ...")
        kept, stats = rollout(model, train_pool, processor, letters, args)
        for s, d in stats.items():
            du.print0(f"  rollout {s:9s} n={d['n']:4d} top1={d['top1_acc']:.3f} pass@{args.k}={d['pass@K']:.3f} "
                      f"p_true={d['mean_p_true']:.3f}")
        if args.weight_by_correct:
            sft_items = [train_pool[i] for i, nc in kept for _ in range(nc)]
        else:
            sft_items = [train_pool[i] for i, _ in kept]
        du.print0(f"  kept {len(kept)}/{len(train_pool)} items, SFT on {len(sft_items)} ...")
        sft = sft_train(model, sft_items, processor, letters, args, args.sft_epochs, args.sft_max_steps,
                        epoch_offset=r * args.sft_epochs, tag=f"  round {r + 1} ")
        ev = eval_all(model, eval_pool, args, processor, letters)
        for s, d in ev.items():
            b = history["before"].get(s, {})
            du.print0(f"  after  {s:9s} n={d['n']:4d} acc={d['acc']:.3f} "
                      f"(before {b.get('acc', float('nan')):.3f}) ece={d['ece']:.4f}")
        history["rounds"].append({"round": r + 1, "kept": len(kept), "sft": sft,
                                  "rollout": stats, "eval": ev})
        save_adapter(model, args.out / f"adapter-round{r + 1}")

    save_adapter(model, args.out / "adapter")
    if du.is_main():
        processor.save_pretrained(args.out / "processor")
        (args.out / "metrics.json").write_text(json.dumps(history, indent=2))
    du.print0("saved ->", args.out)


if __name__ == "__main__":
    main()
