"""One forward pass per Cauldron item: option-letter probabilities + prompt-end hidden states.

Writes one create-only shard per subset: <out>/<subset>.pt

Throughput: the Qwen3.5 linear-attention Triton kernels re-tune for every new input shape,
so images are letterboxed to one square size and sequences padded to a fixed multiple.
A background thread streams data and runs CPU preprocessing while the GPU scores.
"""

import argparse
import queue
import threading
import time
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from cauldron_tasks import CAULDRON_REVISION, LETTERS, SUBSETS, iter_items
from extract_onepass import auto_layers

MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
INSTRUCTION = "Answer with the option letter only."


def load_model(model_id="Qwen/Qwen3.5-4B", revision=MODEL_REVISION):
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    processor = AutoProcessor.from_pretrained(
        model_id, revision=revision, min_pixels=128 * 32 * 32, max_pixels=448 * 32 * 32)
    processor.tokenizer.padding_side = "left"
    return model, processor


def prompt_text(item):
    lines = [f"Question: {item['question']}", "Options:"]
    lines += [f"{LETTERS[i]}. {o}" for i, o in enumerate(item["options"])]
    lines.append(INSTRUCTION)
    return "\n".join(lines)


def letterbox(image, size):
    """Fit inside size x size keeping aspect ratio, pad with neutral gray."""
    scale = size / max(image.size)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                           Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def letter_ids(tokenizer):
    """Bare letter tokens; each must be exactly one round-trip token (no summed subtokens)."""
    ids = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Letter {letter!r} is not a single token")
        ids.append(encoded[0])
    return ids


def prepare(processor, items, image_size, pad_multiple):
    """CPU side: letterbox, chat template, processor. Runs in the prefetch thread."""
    messages = [
        [{"role": "user", "content": [{"type": "image", "image": letterbox(img, image_size)} for img in it["images"]]
          + [{"type": "text", "text": prompt_text(it)}]}]
        for it in items
    ]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
             for m in messages]
    image_inputs, _ = process_vision_info(messages)
    return processor(text=texts, images=image_inputs, return_tensors="pt", padding=True,
                     pad_to_multiple_of=pad_multiple)


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
        logits = model(**inputs).logits[:, -1, :].float()
    finally:
        for h in handles:
            h.remove()
    letter_logits = logits[:, ids].cpu()
    states = torch.stack([captured[l] for l in layers], dim=1).to(torch.float16).cpu()  # [B, L, D]
    return letter_logits, states


def prefetch(subset, args, processor, out_queue):
    """Producer: stream items, group into batches, preprocess, hand over to the GPU loop."""
    try:
        batch = []
        for item in iter_items(subset, args.limit, seed=args.seed):
            batch.append(item)
            if len(batch) == args.batch:
                out_queue.put((batch, prepare(processor, batch, args.image_size, args.pad_multiple)))
                batch = []
        if batch:
            out_queue.put((batch, prepare(processor, batch, args.image_size, args.pad_multiple)))
    except Exception as error:  # surface producer failures in the main thread
        out_queue.put(error)
    out_queue.put(None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="new directory for per-subset shards")
    ap.add_argument("--subsets", nargs="+", default=SUBSETS)
    ap.add_argument("--limit", type=int, default=300, help="items per subset")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=False)
    model, processor = load_model()
    layers = auto_layers(model)
    ids = letter_ids(processor.tokenizer)
    print("layers:", layers, flush=True)

    for subset in args.subsets:
        records, states_all = [], []
        started = time.perf_counter()
        batches = queue.Queue(maxsize=4)
        producer = threading.Thread(target=prefetch, args=(subset, args, processor, batches), daemon=True)
        producer.start()
        pbar = tqdm(total=args.limit, desc=subset)
        while (got := batches.get()) is not None:
            if isinstance(got, Exception):
                raise got
            batch, inputs = got
            logits, states = forward_batch(model, inputs, layers, ids)
            for it, lg, st in zip(batch, logits, states):
                k = len(it["options"])
                probs = torch.softmax(lg[:k], dim=-1)
                pred = int(probs.argmax())
                records.append({
                    "subset": it["subset"], "image_id": it["image_id"], "image_key": it["image_key"],
                    "qa_index": it["qa_index"], "question": it["question"], "options": it["options"],
                    "answer": it["answer"], "answer_kind": it["answer_kind"], "pred": pred,
                    "probs": probs.tolist(), "conf": float(probs[pred]), "correct": int(pred == it["answer"]),
                    "n_images": len(it["images"]),
                })
                states_all.append(st)
            pbar.update(len(batch))
            pbar.set_postfix(acc=round(sum(r["correct"] for r in records) / len(records), 3))
        producer.join()
        pbar.close()
        elapsed = time.perf_counter() - started
        torch.save({
            "model": "Qwen/Qwen3.5-4B", "model_revision": MODEL_REVISION,
            "cauldron_revision": CAULDRON_REVISION, "subset": subset, "layers": layers,
            "prompt_instruction": INSTRUCTION, "image_size": args.image_size, "seed": args.seed,
            "seconds": elapsed, "meta": records, "prompt_states": torch.stack(states_all),
        }, args.out / f"{subset}.pt")
        acc = sum(r["correct"] for r in records) / max(1, len(records))
        print(f"{subset}: {len(records)} items, acc {acc:.3f}, {elapsed:.0f}s "
              f"({len(records) / elapsed:.1f} items/s)", flush=True)


if __name__ == "__main__":
    main()
