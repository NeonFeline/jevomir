"""Shared image prefix vs full-sequence scoring: agreement and speed.

For images with several questions, the full pipeline encodes the image once per question.
Here the prompt is split right after <|vision_end|>: the prefix (template + image) runs once
per image, its cache is repeated for each question, and only the question suffix runs.

Compares, on identical items:
  A  full sequences, batched and left-padded to a multiple of 128 (the extraction path)
  A1 full sequences, one at a time without padding (baseline numerical noise of batching)
  B  shared prefix + right-padded suffix continuation
Reports letter-choice agreement, probability differences, hidden-state cosine, and time.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info

from cauldron_tasks import build_index, decode
from extract_cauldron import forward_batch, letter_ids, letterbox, load_model, prepare, prompt_text
from extract_onepass import auto_layers

VISION_END = "<|vision_end|>"


def full_texts(processor, items, image_size):
    messages = [[{"role": "user", "content": [{"type": "image", "image": letterbox(img, image_size)} for img in it["images"]]
                  + [{"type": "text", "text": prompt_text(it)}]}] for it in items]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for m in messages]
    return messages, texts


def repeat_cache(cache, repeats):
    """Repeat every cached batch row `repeats` times. transformers 5.17 lacks this for
    linear-attention layers, so handle their conv/recurrent states and attention K/V here."""
    for layer in cache.layers:
        for name in ("conv_states", "recurrent_states"):
            states = getattr(layer, name, None)
            if isinstance(states, (list, tuple)):
                for i, t in enumerate(states):
                    if torch.is_tensor(t):
                        states[i] = t.repeat_interleave(repeats, dim=0)
            elif isinstance(states, dict):
                for key, t in states.items():
                    if torch.is_tensor(t):
                        states[key] = t.repeat_interleave(repeats, dim=0)
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if torch.is_tensor(t) and t.numel():
                setattr(layer, name, t.repeat_interleave(repeats, dim=0))


@torch.inference_mode()
def score_prefix(model, processor, groups, layers, ids, image_size):
    """groups: list of (items sharing one image). Each group must have the same question count."""
    per = len(groups[0])
    heads = [g[0] for g in groups]
    messages, texts = full_texts(processor, heads, image_size)
    prefix_texts = [t[:t.index(VISION_END) + len(VISION_END)] for t in texts]
    image_inputs, _ = process_vision_info(messages)
    prefix = processor(text=prefix_texts, images=image_inputs, return_tensors="pt", padding=True)
    if not bool(prefix["attention_mask"].all()):
        raise ValueError("Prefixes differ in length; this benchmark assumes one image size")
    prefix = {k: v.to("cuda") for k, v in prefix.items() if torch.is_tensor(v)}
    P = prefix["input_ids"].shape[1]

    out = model.model(**prefix, use_cache=True)
    cache = out.past_key_values
    repeat_cache(cache, per)

    # Suffixes: every question of every image, in group order, right-padded.
    items = [it for g in groups for it in g]
    _, texts_all = full_texts(processor, items, image_size)
    suffix_texts = [t[t.index(VISION_END) + len(VISION_END):] for t in texts_all]
    tok = processor.tokenizer
    suffix_ids = [tok.encode(s, add_special_tokens=False) for s in suffix_texts]
    S = max(len(s) for s in suffix_ids)
    pad = tok.pad_token_id
    input_ids = torch.full((len(items), S), pad, dtype=torch.long)
    mask = torch.zeros((len(items), P + S), dtype=torch.long)
    mask[:, :P] = 1
    for i, s in enumerate(suffix_ids):
        input_ids[i, :len(s)] = torch.tensor(s)
        mask[i, P:P + len(s)] = 1

    # Exact 3D M-RoPE positions: compute on each full sequence, keep the suffix part.
    prefix_ids = prefix["input_ids"].repeat_interleave(per, dim=0)
    mm = prefix["mm_token_type_ids"].repeat_interleave(per, dim=0)
    grid = prefix["image_grid_thw"].repeat_interleave(per, dim=0)
    positions = torch.zeros((3, len(items), S), dtype=torch.long, device="cuda")
    for i, s in enumerate(suffix_ids):
        full = torch.cat([prefix_ids[i], torch.tensor(s, device="cuda")])[None]
        full_mm = torch.cat([mm[i], torch.zeros(len(s), dtype=mm.dtype, device="cuda")])[None]
        pos, _ = model.model.get_rope_index(full, image_grid_thw=grid[i:i + 1], mm_token_type_ids=full_mm)
        positions[:, i, :len(s)] = pos[:, 0, P:]
        positions[:, i, len(s):] = pos[:, 0, -1:]

    captured = {}

    def make_hook(l):
        def hook(module, args, output):
            captured[l] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = [model.model.language_model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    try:
        hidden = model.model(input_ids=input_ids.cuda(), attention_mask=mask.cuda(), position_ids=positions,
                             past_key_values=cache, use_cache=True).last_hidden_state
    finally:
        for h in handles:
            h.remove()
    last = torch.tensor([len(s) - 1 for s in suffix_ids], device="cuda")
    rows = torch.arange(len(items), device="cuda")
    logits = model.lm_head(hidden[rows, last]).float()[:, ids].cpu()
    states = torch.stack([captured[l][rows, last] for l in layers], dim=1).to(torch.float16).cpu()
    return items, logits, states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path.home() / "cauldron")
    ap.add_argument("--subset", default="clevr")
    ap.add_argument("--images", type=int, default=384)
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--image-batch", type=int, default=32, help="images per prefix batch")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    index = build_index(args.subset, args.data, args.images * args.per_image * 3, seed=args.seed,
                        per_image=args.per_image)
    by_image = defaultdict(list)
    for it in index:
        by_image[it["image_id"]].append(it)
    groups = [g for g in by_image.values() if len(g) == args.per_image][:args.images]
    groups = [[decode(it) for it in g] for g in groups]
    items = [it for g in groups for it in g]
    print(f"{len(groups)} images x {args.per_image} questions = {len(items)} items", flush=True)

    model, processor = load_model()
    layers, ids = auto_layers(model), letter_ids(processor.tokenizer)
    full_batch = args.image_batch * args.per_image

    def run_full(bs, pad_multiple):
        logits, states = [], []
        for i in range(0, len(items), bs):
            inputs = prepare(processor, items[i:i + bs], args.image_size, pad_multiple)
            lg, st = forward_batch(model, dict(inputs), layers, ids)
            logits.append(lg), states.append(st)
        return torch.cat(logits), torch.cat(states)

    def run_prefix():
        logits, states = [], []
        for i in range(0, len(groups), args.image_batch):
            _, lg, st = score_prefix(model, processor, groups[i:i + args.image_batch], layers, ids, args.image_size)
            logits.append(lg), states.append(st)
        return torch.cat(logits), torch.cat(states)

    def timed(fn, *a):
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = fn(*a)
        torch.cuda.synchronize()
        return result, time.perf_counter() - t

    # Warm the kernels for each path once on a small slice, then time the full runs.
    run_full(full_batch, 128), run_prefix()
    (la, sa), ta = timed(run_full, full_batch, 128)
    (lb, sb), tb = timed(run_prefix)
    la1, sa1 = run_full(1, None)

    k = torch.tensor([len(it["options"]) for it in items])

    def probs(logits):
        out = torch.zeros_like(logits)
        for i, n in enumerate(k.tolist()):
            out[i, :n] = torch.softmax(logits[i, :n], dim=-1)
        return out

    pa, pa1, pb = probs(la), probs(la1), probs(lb)

    def compare(name, p, s, ref_p, ref_s):
        agree = (p.argmax(-1) == ref_p.argmax(-1)).float().mean().item()
        diff = (p - ref_p).abs().max(-1).values
        cos = F.cosine_similarity(s.float(), ref_s.float(), dim=-1)  # [N, L]
        print(f"{name}: argmax agreement {agree:.4f} | max|dp| mean {diff.mean():.2e} p99 {diff.quantile(0.99):.2e} "
              f"max {diff.max():.2e} | hidden cos min per layer "
              + " ".join(f"{c:.5f}" for c in cos.min(0).values.tolist()), flush=True)

    compare("A1 (bs=1, no pad) vs A (batched)", pa1, sa1, pa, sa)
    compare("B  (shared prefix) vs A (batched)", pb, sb, pa, sa)
    compare("B  (shared prefix) vs A1 (bs=1)  ", pb, sb, pa1, sa1)
    acc = lambda p: (p.argmax(-1) == torch.tensor([it["answer"] for it in items])).float().mean().item()
    print(f"accuracy A {acc(pa):.4f} A1 {acc(pa1):.4f} B {acc(pb):.4f}", flush=True)
    print(f"time A {ta:.1f}s ({len(items)/ta:.1f} items/s) | B {tb:.1f}s ({len(items)/tb:.1f} items/s) "
          f"| speedup {ta/tb:.2f}x (GPU only, preprocessing included in both)", flush=True)


if __name__ == "__main__":
    main()
