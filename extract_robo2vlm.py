"""One forward pass per Robo2VLM-1 item: letter probabilities + prompt-end hidden states.

Reads the cleaned index produced by robo2vlm_tasks.py and the source parquet shards,
letterboxes images to 448 (same as Cauldron) and writes a Cauldron-compatible shard
<out>/robo2vlm-<stem>.pt that train_probe_cauldron.py can consume.

CLI: python extract_robo2vlm.py --index runs/agentic/robo2vlm_test-00000-of-00003.index.jsonl \
        --parquet ~/agentic/robo2vlm/data/test-00000-of-00003.parquet \
        --out runs/robo2vlm-features --limit 0
"""

import argparse
import io
import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm.auto import tqdm

from extract_cauldron import INSTRUCTION, forward_batch, letter_ids, load_model, prepare
from extract_onepass import auto_layers
from review_datasets import load_jsonl


def image_bytes_by_id(parquet_path):
    rows = pq.read_table(parquet_path, columns=["id", "image"]).to_pylist()
    out = {}
    for r in rows:
        img = r.get("image") or {}
        if img.get("bytes"):
            out[r["id"]] = img["bytes"]
    return out


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", nargs="+", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    images_by_id = image_bytes_by_id(args.parquet)
    model, processor = load_model()
    layers = auto_layers(model)
    ids = letter_ids(processor.tokenizer)
    print("layers:", layers, "| images in parquet:", len(images_by_id), flush=True)

    for index_path in args.index:
        index = load_jsonl(index_path)
        items = []
        for it in index:
            data = images_by_id.get(it["item_id"])
            if not data:
                continue
            items.append({**it, "images": [Image.open(io.BytesIO(data)).convert("RGB")]})
        if args.limit:
            items = items[:args.limit]
        if not items:
            print(f"{index_path.stem}: no matching items", flush=True)
            continue
        records, states_all = [], []
        t0 = time.perf_counter()
        pbar = tqdm(total=len(items), desc=index_path.stem, mininterval=10)
        for start in range(0, len(items), args.batch):
            batch = items[start:start + args.batch]
            inputs = prepare(processor, batch, args.image_size, args.pad_multiple)
            logits, states = forward_batch(model, inputs, layers, ids)
            for it, lg, st in zip(batch, logits, states):
                k = len(it["options"])
                probs = torch.softmax(lg[:k], dim=-1)
                pred = int(probs.argmax())
                records.append({
                    "subset": it["subset"], "image_id": it["item_id"], "image_key": it["image_key"],
                    "qa_index": 0, "question": it["question"], "options": it["options"],
                    "answer": int(it["answer"]), "answer_kind": it["answer_kind"], "pred": pred,
                    "probs": probs.tolist(), "conf": float(probs[pred]),
                    "correct": int(pred == int(it["answer"])), "n_images": len(it["images"]),
                })
                states_all.append(st)
            pbar.update(len(batch))
        pbar.close()
        acc = sum(r["correct"] for r in records) / max(1, len(records))
        stem = args.parquet.stem if len(args.index) == 1 else index_path.stem
        torch.save({"model": "Qwen/Qwen3.5-4B", "subset": stem, "layers": layers,
                    "prompt_instruction": INSTRUCTION, "meta": records,
                    "prompt_states": torch.stack(states_all)}, args.out / f"{stem}.pt")
        print(f"{stem}: {len(records)} items, acc {acc:.3f}, {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
