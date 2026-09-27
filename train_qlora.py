"""QLoRA SFT for the one-pass letter reader.

Training objective == inference objective: one forward pass over the prompt
(image + question + options + instruction + assistant scaffold). The loss is
computed ONLY on the final position, whose target is the answer-letter token.
No prompt token contributes to the loss, so the model cannot "learn the prompt".

Options are optionally shuffled per epoch (the answer index follows) to reduce
the position bias noted in API.md. Evaluates accuracy / raw ECE per subset
before and after training.

Single GPU:  python train_qlora.py --out runs/qlora-001
8 GPUs:      torchrun --standalone --nproc_per_node 8 train_qlora.py --out runs/qlora-001
             (or bash run_train_8gpu.sh qlora --out runs/qlora-001)
The global batch is batch * grad_accum * world_size; eval is sharded over ranks.
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
from torch.utils.data.distributed import DistributedSampler

import bitsandbytes as bnb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

import dist_utils as du
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
TEST_FRAC, VAL_FRAC = 0.15, 0.1


def parse_sizes(spec):
    out = []
    for part in spec.split(","):
        name, _, limit = part.partition(":")
        out.append((name.strip(), int(limit)))
    return out


def load_pool(data_dir, subsets, want, seed, test_frac=TEST_FRAC, val_frac=VAL_FRAC):
    """Items whose perceptual image hash falls in the requested split bucket.

    Deterministic, so every rank builds the identical pool without communication. The
    candidate budget is scaled by the bucket's share (test is only 15% of items) and doubled
    until `need` items are found or the subset runs out.
    """
    frac = {"test": test_frac, "val": val_frac}.get(want, 1 - test_frac - val_frac)
    pool = []
    for subset, need in subsets:
        limit, keys = int(need / frac * 1.3) + 50, {}
        while True:
            candidates = build_index(subset, Path(data_dir), limit, seed=seed)
            picked = []
            for it in candidates:
                k = keys.get(it["image_id"])
                if k is None:
                    k = keys[it["image_id"]] = image_key(
                        [Image.open(io.BytesIO(b)).convert("RGB") for b in it["image_bytes"]])
                if split_of(k, test_frac, val_frac) == want:
                    picked.append((it, k))
                    if len(picked) >= need:
                        break
            if len(picked) >= need or len(candidates) < limit:
                break
            limit *= 2
        for it, k in picked:
            pool.append({
                "subset": subset, "image_id": it["image_id"], "qa_index": it.get("qa_index", 0),
                "image_key": k, "question": it["question"], "options": list(it["options"]),
                "answer": it["answer"], "answer_kind": it["answer_kind"],
                "image_bytes": it["image_bytes"],
            })
        short = "" if len(picked) >= need else f"  (WANTED {need}: subset exhausted)"
        du.print0(f"  pool[{want}] {subset}: {len(picked)} items{short}")
    return pool


def load_train_pool(args, spec):
    """Train pool, optionally skipping the first N items per subset of the same seeded order.

    With --skip-train-subsets set to an earlier stage's --train-subsets (same seed), this
    returns items that stage never trained on; images of skipped items are excluded too, so
    no picture is shared across stages. Pool order is prefix-stable in `need`, which makes
    "first N" well defined.
    """
    skip = dict(parse_sizes(args.skip_train_subsets)) if getattr(args, "skip_train_subsets", None) else {}
    if not skip:
        return load_pool(args.data, spec, "train", args.seed)
    pool = []
    for subset, need in spec:
        k = skip.get(subset, 0)
        items = load_pool(args.data, [(subset, k + need)], "train", args.seed)
        used = {it["image_id"] for it in items[:k]}
        fresh = [it for it in items[k:] if it["image_id"] not in used][:need]
        du.print0(f"  fresh[train] {subset}: {len(fresh)} items after skipping the first {k}")
        pool += fresh
    return pool


def item_seed(it, salt):
    """Stable per-question seed (image_id alone is shared by up to 3 questions per image)."""
    key = f"{it['image_id']}#{it.get('qa_index', 0)}#{salt}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


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
            rng = random.Random(item_seed(it, f"epoch{self.epoch}"))
            order = list(range(len(options)))
            rng.shuffle(order)
            options = [options[j] for j in order]
            answer = order.index(answer)
        return {
            "index": i, "images": [Image.open(io.BytesIO(b)).convert("RGB") for b in it["image_bytes"]],
            "question": it["question"], "options": options, "answer": answer,
            "subset": it["subset"], "answer_kind": it["answer_kind"], "image_key": it["image_key"],
            "image_id": it["image_id"], "qa_index": it.get("qa_index", 0),
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


def trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def build_model(args):
    """(Q)LoRA policy on this rank's GPU; LoRA init is broadcast from rank 0."""
    device = torch.cuda.current_device()
    quant = None if args.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    # Explicit device_map: without it --no-4bit left the model on the CPU, and 4-bit would
    # put every rank's copy on cuda:0.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, quantization_config=quant,
        dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device})
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if getattr(args, "init_adapter", None):
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=args.targets))
    du.broadcast_params(trainable(model))
    if du.is_main():
        model.print_trainable_parameters()
    return model


def load_processor(args):
    du.fetch_model(args.model, args.revision)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision,
                                              min_pixels=128 * 32 * 32, max_pixels=448 * 32 * 32)
    processor.tokenizer.padding_side = "left"
    return processor, letter_ids(processor.tokenizer)


def lr_schedule(peak, warmup, total_steps):
    """Linear warmup then cosine. Warmup is capped at 20% of the run: on 8 GPUs a small pool
    gives few optimizer steps, and a fixed warmup of 20 could otherwise cover the whole run."""
    warmup = min(warmup, max(1, total_steps // 5))

    def lr_at(step):
        if step < warmup:
            return peak * (step + 1) / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return lr_at, warmup


def optimizer_step(model, optimizer, max_norm=1.0):
    """All-reduce grads, clip, step. Skips the update (on every rank alike, since the synced
    grads are identical) when the gradient norm is not finite. Returns (norm, stepped)."""
    params = trainable(model)
    du.sync_grads(params)
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    ok = bool(torch.isfinite(norm))
    if ok:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm), ok


def make_loader(dataset, batch, workers, collate, seed, shuffle, drop_last):
    sampler = DistributedSampler(dataset, num_replicas=du.world_size(), rank=du.rank(),
                                 shuffle=shuffle, seed=seed, drop_last=drop_last)
    return DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=workers,
                      collate_fn=collate, pin_memory=True, drop_last=drop_last,
                      prefetch_factor=4 if workers else None,
                      persistent_workers=False), sampler


def batches_per_rank(n_items, batch):
    return (n_items // du.world_size()) // batch


def save_adapter(model, path):
    if du.is_main():
        model.save_pretrained(path)
    du.barrier()


def sft_train(model, items, processor, letters, args, epochs, max_steps=0, epoch_offset=0, tag="",
              ckpt_dir=None):
    """Letter-token SFT, data-parallel over ranks. Every rank runs the same number of batches
    (DistributedSampler + drop_last), so the per-step all-reduce never deadlocks."""
    per_rank = batches_per_rank(len(items), args.batch)
    steps_per_epoch = per_rank // args.grad_accum
    if steps_per_epoch == 0:
        du.print0(f"{tag}skip SFT: {len(items)} items < one global batch "
                  f"({args.batch} x {args.grad_accum} x {du.world_size()} ranks)")
        return {"steps": 0, "items": len(items), "skipped_nonfinite": 0}
    total_steps = min(max_steps, steps_per_epoch * epochs) if max_steps else steps_per_epoch * epochs
    lr_at, warmup = lr_schedule(args.lr, args.warmup, total_steps)
    du.print0(f"{tag}SFT: {len(items)} items, global batch {args.batch * args.grad_accum * du.world_size()}, "
              f"{total_steps} steps, warmup {warmup}")
    optimizer = bnb.optim.PagedAdamW8bit(trainable(model), lr=args.lr, weight_decay=0.0)

    model.train()
    step, seen, micro, skipped, last_log = 0, 0, 0, 0, 0
    log = {"loss": 0.0, "n": 0.0}
    t0, done = time.perf_counter(), False
    deadline = t0 + args.time_budget_min * 60 if getattr(args, "time_budget_min", 0) else None
    save_every = getattr(args, "save_every", 0)
    for epoch in range(epochs):
        dataset = ItemDataset(items, shuffle_options=args.shuffle_options, epoch=epoch_offset + epoch)
        loader, sampler = make_loader(dataset, args.batch, args.workers,
                                      Collate(processor, letters, args.image_size, args.pad_multiple),
                                      args.seed, shuffle=True, drop_last=True)
        sampler.set_epoch(epoch_offset + epoch)
        # Only whole accumulation groups, so no partial-group gradient leaks into the next epoch.
        usable = (len(loader) // args.grad_accum) * args.grad_accum
        for b_idx, (meta, inputs, answer) in enumerate(loader):
            if b_idx >= usable:
                break
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :]
                loss = F.cross_entropy(logits.float(), answer.cuda())
            (loss / args.grad_accum).backward()
            log["loss"] += loss.item() * len(answer)
            log["n"] += len(answer)
            micro += 1
            seen += len(answer)
            if micro % args.grad_accum:
                continue
            norm, ok = optimizer_step(model, optimizer)
            skipped += int(not ok)
            step += 1
            if step % args.log_every == 0 or step == 1 or step == total_steps:
                tot = du.all_reduce_sum(log)
                du.print0(f"{tag}step {step:5d}/{total_steps} loss {tot['loss'] / max(1, tot['n']):.4f} "
                          f"gnorm {norm:.3f} lr {lr_at(step - 1):.2e} seen {seen * du.world_size()} "
                          f"{(time.perf_counter() - t0) / (step - last_log):.2f}s/step"
                          + ("" if ok else " NONFINITE-SKIPPED"))
                t0, last_log, log = time.perf_counter(), step, {"loss": 0.0, "n": 0.0}
            if ckpt_dir and save_every and step % save_every == 0 and step < total_steps:
                save_adapter(model, Path(ckpt_dir) / f"checkpoint-{step}")
            if step >= total_steps:
                done = True
                break
            if deadline and du.any_true(time.perf_counter() > deadline):
                du.print0(f"{tag}time budget reached at step {step}/{total_steps}")
                done = True
                break
        if done:
            break
    return {"steps": step, "planned_steps": total_steps, "items": len(items), "skipped_nonfinite": skipped}


@torch.inference_mode()
def evaluate(model, items, processor, letter_token_ids, batch, workers, image_size, pad_multiple):
    """Per-subset accuracy, confidences and correctness, sharded over ranks and gathered."""
    per_subset = {}
    letter_index = {tid: i for i, tid in enumerate(letter_token_ids)}
    mine = du.shard(items)
    was_training = model.training
    model.eval()
    try:
        if mine:
            loader = DataLoader(ItemDataset(mine), batch_size=batch, num_workers=workers,
                                collate_fn=Collate(processor, letter_token_ids, image_size, pad_multiple),
                                pin_memory=True)
            for meta, inputs, answer in loader:
                logits = model(**move(inputs, "cuda"), logits_to_keep=1).logits[:, -1, :].float().cpu()
                letters = logits[:, letter_token_ids]
                for j, m in enumerate(meta):
                    probs = torch.softmax(letters[j, :m["n_options"]], dim=-1)
                    correct = int(int(probs.argmax()) == letter_index[int(answer[j])])
                    d = per_subset.setdefault(m["subset"], {"n": 0, "correct": 0, "confs": [], "labels": [],
                                                            "kinds": {}})
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
    merged = {}
    for part in du.gather(per_subset):
        for sub, d in part.items():
            m = merged.setdefault(sub, {"n": 0, "correct": 0, "confs": [], "labels": [], "kinds": {}})
            m["n"] += d["n"]
            m["correct"] += d["correct"]
            m["confs"] += d["confs"]
            m["labels"] += d["labels"]
            for k, v in d["kinds"].items():
                mk = m["kinds"].setdefault(k, {"n": 0, "correct": 0})
                mk["n"] += v["n"]
                mk["correct"] += v["correct"]
    return merged


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
    if not pool:
        return {}
    raw = evaluate(model, pool, processor, letters, args.eval_batch,
                   args.eval_workers, args.image_size, args.pad_multiple)
    return dict(sorted(summarize(raw).items()))


def seed_everything(seed):
    # Same seed everywhere for anything that must agree (pools, LoRA init is broadcast anyway);
    # per-rank streams for sampling are derived with seed_rank().
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def seed_rank(seed):
    s = seed * 1000 + du.rank()
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def add_common_args(ap):
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("JEV_DATA", Path.home() / "cauldron")))
    ap.add_argument("--train-subsets", default="tallyqa:400,nlvr2:400,iconqa:300,clevr:200,vqav2:200")
    ap.add_argument("--eval-subsets", default="tallyqa:150,nlvr2:150,iconqa:150,clevr:100,vqav2:100")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--init-adapter", type=Path, default=None, help="start from an SFT/RFT adapter")
    ap.add_argument("--skip-train-subsets", default="",
                    help="skip the first N train items per subset (an earlier stage's --train-subsets)")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4, help="per-GPU micro-batch")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--shuffle-options", action="store_true")
    ap.add_argument("--no-4bit", action="store_true", help="bf16 LoRA (faster on 80GB cards)")
    ap.add_argument("--workers", type=int, default=8, help="DataLoader workers per GPU")
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=0, help="adapter checkpoint every N steps (0=off)")
    ap.add_argument("--time-budget-min", type=float, default=0,
                    help="stop training (then eval + save as usual) after this many minutes of steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")


def prepare_out(out):
    """Refuse to silently mix a new run into an old one's adapter/metrics."""
    if du.is_main():
        if (out / "metrics.json").exists():
            raise SystemExit(f"{out}/metrics.json exists; pick a new --out")
        out.mkdir(parents=True, exist_ok=True)
    du.barrier()


def jsonable_args(args):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="caps epochs when > 0")
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
        n = 8 * du.world_size()
        train_spec, eval_spec = [(s, n) for s, _ in train_spec[:2]], [(s, n) for s, _ in eval_spec[:2]]
        args.max_steps = args.max_steps or 5
        args.batch, args.grad_accum, args.workers, args.eval_workers = 2, 1, 2, 2
        args.epochs = 1

    du.print0(f"world size {du.world_size()}; building pools ...")
    train_pool = load_pool(args.data, train_spec, "train", args.seed)
    eval_pool = load_pool(args.data, eval_spec, "test", args.seed)
    if not train_pool:
        raise SystemExit("no training items found; check --data and --train-subsets")

    processor, letters = load_processor(args)
    model = build_model(args)
    seed_rank(args.seed)

    before = {}
    if eval_pool:
        du.print0("eval before ...")
        before = eval_all(model, eval_pool, args, processor, letters)
        for s, d in before.items():
            du.print0(f"  before {s:9s} n={d['n']:4d} acc={d['acc']:.3f} conf={d['mean_conf']:.3f} "
                      f"ece={d['ece']:.4f}")

    train = sft_train(model, train_pool, processor, letters, args, args.epochs, args.max_steps,
                      ckpt_dir=args.out)

    du.print0("eval after ...")
    after = eval_all(model, eval_pool, args, processor, letters)
    for s, d in after.items():
        b = before.get(s, {})
        du.print0(f"  after  {s:9s} n={d['n']:4d} acc={d['acc']:.3f} (before {b.get('acc', float('nan')):.3f}) "
                  f"conf={d['mean_conf']:.3f} ece={d['ece']:.4f}")

    save_adapter(model, args.out / "adapter")
    if du.is_main():
        processor.save_pretrained(args.out / "processor")
        (args.out / "metrics.json").write_text(json.dumps({
            "train_items": len(train_pool), "eval_items": len(eval_pool), **train,
            "world_size": du.world_size(), "global_batch": args.batch * args.grad_accum * du.world_size(),
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                     "targets": args.targets, "4bit": not args.no_4bit},
            "shuffle_options": args.shuffle_options, "before": before, "after": after,
            "args": jsonable_args(args),
        }, indent=2))
    du.print0("saved adapter ->", args.out / "adapter")


if __name__ == "__main__":
    main()
