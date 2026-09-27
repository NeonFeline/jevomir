"""One forward pass per (image, protocol). No token generation.

Forced-choice protocols: the model's answer distribution is read from the logits
at the final prompt position. Candidate probability = softmax mass summed over
both surface forms ("cat" / " cat"). Prompt-end hidden states from several LLM
layers feed the calibration probe. Raw confidence = probability of the model's
argmax choice.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import ImageFilter
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "Qwen/Qwen3.5-4B"

PROTOCOLS = [
    {"id": 0, "question": "Does this image contain a cat? Answer with one word: cat or dog.",
     "pos": "cat", "neg": "dog"},
    {"id": 1, "question": "Which animal is in this photo, a cat or a dog? Answer with one word.",
     "pos": "cat", "neg": "dog"},
    {"id": 2, "question": "Does this image contain a cat? Answer yes or no.",
     "pos": "yes", "neg": "no"},
    {"id": 3, "question": "Is there a cat in this picture? Answer yes or no.",
     "pos": "yes", "neg": "no"},
]


def cand_ids(tokenizer, word):
    """Next-token ids for both surface forms. Each form must be one token: summing the
    first-position mass of every subtoken of a multi-token word is not P(word)."""
    ids = []
    for form in (word, " " + word):
        encoded = tokenizer.encode(form, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Candidate {form!r} is {len(encoded)} tokens; one-pass readout needs one")
        ids.append(encoded[0])
    return ids


def auto_layers(model):
    n = getattr(model.config, "text_config", model.config).num_hidden_layers
    return sorted(set(int(round(x)) for x in np.linspace(0.1 * (n - 1), n - 2, 6)))


def load_model(model_id=MODEL, device="cuda"):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to(device).eval()
    try:
        processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=128 * 32 * 32, max_pixels=448 * 32 * 32
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


@torch.inference_mode()
def forward_batch(model, processor, images, question, layers, ids_pos, ids_neg):
    """Score one prompt per batch. `images` may contain None entries -> text-only prompt."""
    messages = []
    for img in images:
        content = []
        if img is not None:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})
        messages.append([{"role": "user", "content": content}])
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for m in messages
    ]
    has_images = any(img is not None for img in images)
    if has_images:
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=texts, return_tensors="pt", padding=True)
    inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}

    captured = {}

    def make_hook(l):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[l] = hs[:, -1, :].detach().float().cpu()
        return hook

    handles = [model.model.language_model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        out = model(**inputs)
    finally:
        for h in handles:
            h.remove()

    probs = torch.softmax(out.logits[:, -1, :].float(), dim=-1)
    p_pos = probs[:, ids_pos].sum(-1)
    p_neg = probs[:, ids_neg].sum(-1)
    p = p_pos / (p_pos + p_neg).clamp(min=1e-9)
    states = torch.stack([captured[l] for l in layers], dim=1)  # [B, L, D]
    return p.tolist(), states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blur", type=float, default=0.0)
    args = ap.parse_args()

    ds = load_dataset("timm/oxford-iiit-pet")[args.split]
    n = min(args.limit, len(ds))
    order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed)).tolist()
    idxs = order[:n]
    radii = (dict(zip(idxs, np.random.default_rng(args.seed + 1).uniform(1.0, args.blur, size=n)))
             if args.blur > 0 else {i: 0.0 for i in idxs})
    # sort by pixel area so batches have near-equal prompt lengths (less padding)
    idxs.sort(key=lambda i: ds[i]["image"].size[0] * ds[i]["image"].size[1])

    model, processor = load_model(args.model)
    if args.layers is None:
        args.layers = auto_layers(model)
    print("model:", args.model, "| layers:", args.layers)
    tok = processor.tokenizer
    ids = {w: cand_ids(tok, w) for w in ("cat", "dog", "yes", "no")}

    records = []
    pbar = tqdm(total=((n + args.batch - 1) // args.batch) * len(PROTOCOLS), desc=args.split)
    for proto in PROTOCOLS:
        for start in range(0, n, args.batch):
            batch_idx = idxs[start:start + args.batch]
            images = [ds[i]["image"].convert("RGB") for i in batch_idx]
            if args.blur > 0:
                images = [img.filter(ImageFilter.GaussianBlur(radius=float(radii[i]))) for img, i in zip(images, batch_idx)]
            p_vals, states = forward_batch(model, processor, images, proto["question"], args.layers,
                                           ids[proto["pos"]], ids[proto["neg"]])
            for j, i in enumerate(batch_idx):
                true_species = ds.features["label_cat_dog"].names[ds[i]["label_cat_dog"]]
                p_cat = p_vals[j]
                verdict_cat = p_cat >= 0.5
                conf = p_cat if verdict_cat else 1.0 - p_cat
                correct = int(verdict_cat == (true_species == "cat"))
                records.append({
                    "image_id": ds[i]["image_id"],
                    "breed": ds.features["label"].names[ds[i]["label"]],
                    "true_species": true_species,
                    "protocol_id": proto["id"],
                    "question": proto["question"],
                    "answer": proto["pos"] if verdict_cat else proto["neg"],
                    "p_cat": p_cat,
                    "verdict_cat": int(verdict_cat),
                    "conf": conf,
                    "correct": correct,
                    "prompt_states": states[j].to(torch.float16),
                })
            pbar.update(1)
            pbar.set_postfix(acc=round(sum(r["correct"] for r in records) / len(records), 3),
                             conf=round(sum(r["conf"] for r in records) / len(records), 3))

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = [{k: v for k, v in r.items() if k != "prompt_states"} for r in records]
    payload = {
        "protocols": PROTOCOLS,
        "layers": args.layers,
        "model": args.model,
        "split": args.split,
        "meta": meta,
        "prompt_states": torch.stack([r["prompt_states"] for r in records]),
    }
    torch.save(payload, out_path)
    print(f"saved {len(records)} one-pass records -> {out_path}")


if __name__ == "__main__":
    main()
