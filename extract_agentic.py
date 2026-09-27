"""One forward pass per text-only agentic item: option-letter probabilities + prompt-end hidden states.

Writes a Cauldron-compatible shard <out>/<subset>.pt, so train_probe_cauldron.py can train
the calibration probe on agentic items (alone or mixed with image shards). Prompts are
left-padded and padded to a multiple of 128 (the Qwen3.5 linear-attention kernels re-tune
on every new shape).

CLI: python extract_agentic.py --items runs/agentic/bfcl_selection.jsonl --out runs/agentic-features
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from cauldron_tasks import LETTERS
from extract_cauldron import INSTRUCTION, letter_ids
from extract_onepass import auto_layers
from review_datasets import load_jsonl

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def item_key(item):
    return hashlib.sha256(item["item_id"].encode()).hexdigest()[:16]


def prompt_text(item):
    lines = [f"Question: {item['question']}", "Options:"]
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"])]
    lines.append(INSTRUCTION)
    return "\n".join(lines)


class ItemDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class Collate:
    def __init__(self, processor, pad_multiple):
        self.processor, self.pad_multiple = processor, pad_multiple

    def __call__(self, batch):
        messages = [[{"role": "user", "content": [{"type": "text", "text": prompt_text(it)}]}] for it in batch]
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False) for m in messages]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True,
                                pad_to_multiple_of=self.pad_multiple)
        return batch, dict(inputs)


def load_model(model_id=MODEL_ID, revision=MODEL_REVISION):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    processor.tokenizer.padding_side = "left"
    return model, processor


@torch.inference_mode()
def forward_batch(model, inputs, layers, ids):
    inputs = {k: (v.to("cuda", non_blocking=True) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    captured = {}

    def make_hook(l):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[l] = hs[:, -1, :].detach()
        return hook

    handles = [model.model.language_model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        logits = model(**inputs, logits_to_keep=1).logits[:, -1, :].float()
    finally:
        for h in handles:
            h.remove()
    letter_logits = logits[:, ids].cpu()
    states = torch.stack([captured[l] for l in layers], dim=1).to(torch.float16).cpu()
    return letter_logits, states


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, processor = load_model()
    layers = auto_layers(model)
    ids = letter_ids(processor.tokenizer)
    print("layers:", layers, flush=True)

    for path in args.items:
        items = load_jsonl(path)
        if args.limit:
            items = items[:args.limit]
        if not items:
            print(f"{path.stem}: no items", flush=True)
            continue
        loader = DataLoader(ItemDataset(items), batch_size=args.batch, shuffle=False,
                            collate_fn=Collate(processor, args.pad_multiple))
        records, states_all = [], []
        for batch, inputs in tqdm(loader, desc=path.stem, mininterval=5):
            logits, states = forward_batch(model, inputs, layers, ids)
            for it, lg, st in zip(batch, logits, states):
                k = len(it["options"])
                probs = torch.softmax(lg[:k], dim=-1)
                pred = int(probs.argmax())
                records.append({
                    "subset": it["subset"], "image_id": it["item_id"], "image_key": item_key(it),
                    "qa_index": 0, "question": it["question"], "options": it["options"],
                    "answer": int(it["answer"]), "answer_kind": it["answer_kind"], "pred": pred,
                    "probs": probs.tolist(), "conf": float(probs[pred]),
                    "correct": int(pred == int(it["answer"])), "n_images": 0,
                })
                states_all.append(st)
        acc = sum(r["correct"] for r in records) / max(1, len(records))
        torch.save({
            "model": MODEL_ID, "model_revision": MODEL_REVISION, "subset": path.stem, "layers": layers,
            "prompt_instruction": INSTRUCTION, "meta": records,
            "prompt_states": torch.stack(states_all),
        }, args.out / f"{path.stem}.pt")
        print(f"{path.stem}: {len(records)} items, acc {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
