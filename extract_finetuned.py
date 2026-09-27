"""Confidence-head features from a fine-tuned (LoRA) answer model, data-parallel over GPUs.

Merges --adapter into the base model, saves the merged model (--save-model, for
api_server.py --model), then runs one forward pass per item and writes the option-letter
probabilities + prompt-end hidden states in the shard format train_probe_cauldron.py reads.

The head must be trained on items the answer model did NOT train on: on its own training
items the model is ~99% right and the head would learn that confidence. So the probe's
train split comes from --train-subsets after --skip-train-subsets (the SFT/RFT items),
its val split from the val bucket and its test split from the usual test pool; all three
are chosen by the same image-hash buckets train_probe_cauldron.py uses to split.

    bash run_train_8gpu.sh extract --adapter runs/X/rft/adapter-round1 --out runs/X/head-features \
        --save-model runs/X/final-rft1/model --skip-train-subsets "tallyqa:8000,..."
"""

import argparse
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader

import dist_utils as du
from extract_cauldron import MODEL_REVISION, INSTRUCTION, forward_batch, letter_ids, load_model, prepare
from extract_onepass import auto_layers
from train_qlora import ItemDataset, load_pool, load_train_pool, parse_sizes


class Collate:
    def __init__(self, processor, image_size, pad_multiple):
        self.processor, self.image_size, self.pad_multiple = processor, image_size, pad_multiple

    def __call__(self, batch):
        return batch, dict(prepare(self.processor, batch, self.image_size, self.pad_multiple))


@torch.inference_mode()
def extract(model, processor, items, layers, ids, args, split):
    """This rank's share of items -> (records, states [N, L, D] fp16)."""
    records, states = [], []
    if not items:
        return records, states
    loader = DataLoader(ItemDataset(items), batch_size=args.batch, num_workers=args.workers,
                        collate_fn=Collate(processor, args.image_size, args.pad_multiple), pin_memory=True)
    for batch, inputs in loader:
        logits, st = forward_batch(model, inputs, layers, ids)
        for it, lg, s in zip(batch, logits, st):
            k = len(it["options"])
            probs = torch.softmax(lg[:k], dim=-1)
            pred = int(probs.argmax())
            records.append({
                "subset": it["subset"], "image_id": it["image_id"], "image_key": it["image_key"],
                "qa_index": it["qa_index"], "question": it["question"], "options": it["options"],
                "answer": it["answer"], "answer_kind": it["answer_kind"], "pred": pred,
                "probs": probs.tolist(), "conf": float(probs[pred]), "correct": int(pred == it["answer"]),
                "n_images": len(it["images"]), "pool": split,
            })
            states.append(s)
    return records, states


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="new directory for feature shards")
    ap.add_argument("--save-model", type=Path, default=None, help="also save the merged bf16 model here")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("JEV_DATA", Path.home() / "cauldron")))
    ap.add_argument("--train-subsets", default="tallyqa:4000,nlvr2:4000,iconqa:3000,clevr:2500,vqav2:2500")
    ap.add_argument("--skip-train-subsets", default="")
    ap.add_argument("--val-subsets", default="tallyqa:600,nlvr2:600,iconqa:500,clevr:400,vqav2:400")
    ap.add_argument("--eval-subsets", default="tallyqa:300,nlvr2:300,iconqa:300,clevr:300,vqav2:300")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    du.init()
    try:
        run(args)
    finally:
        du.cleanup()


def run(args):
    if du.is_main():
        args.out.mkdir(parents=True, exist_ok=False)
    du.barrier()
    du.print0(f"world size {du.world_size()}; building pools ...")
    pools = {"train": load_train_pool(args, parse_sizes(args.train_subsets)),
             "val": load_pool(args.data, parse_sizes(args.val_subsets), "val", args.seed),
             "test": load_pool(args.data, parse_sizes(args.eval_subsets), "test", args.seed)}

    model, processor = load_model()
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload().eval()
    if args.save_model and du.is_main():
        args.save_model.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_model)
        processor.save_pretrained(args.save_model)
        (args.save_model / "SOURCE.txt").write_text(
            f"Qwen/Qwen3.5-4B@{MODEL_REVISION} + LoRA {args.adapter} (merged, bf16)\n")
        du.print0(f"merged model -> {args.save_model}")
    layers = auto_layers(model)
    ids = letter_ids(processor.tokenizer)
    du.print0(f"layers: {layers}")

    for split, pool in pools.items():
        t0 = time.perf_counter()
        records, states = extract(model, processor, du.shard(pool), layers, ids, args, split)
        # One shard per (split, rank); train_probe_cauldron.py concatenates every *.pt it finds.
        torch.save({"model": str(args.adapter), "model_revision": MODEL_REVISION, "layers": layers,
                    "prompt_instruction": INSTRUCTION, "image_size": args.image_size, "seed": args.seed,
                    "meta": records, "prompt_states": torch.stack(states) if states else
                    torch.empty(0, len(layers), model.config.text_config.hidden_size, dtype=torch.float16)},
                   args.out / f"{split}-rank{du.rank()}.pt")
        tot = du.all_reduce_sum({"n": len(records), "correct": sum(r["correct"] for r in records)})
        du.print0(f"{split}: {int(tot['n'])} items, acc {tot['correct'] / max(1, tot['n']):.3f}, "
                  f"{time.perf_counter() - t0:.0f}s")
    du.barrier()
    du.print0("features ->", args.out)


if __name__ == "__main__":
    main()
