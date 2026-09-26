"""Extract VLM answers, token logprobs and multi-layer hidden states for calibration.

For every (image, question) pair we store:
  - generated answer text
  - per-token logprobs (from generation scores) -> jev prob = exp(mean(log p_i))
  - mean-pooled hidden states of the answer tokens at several LLM layers
  - hidden states of the last prompt token (image+question summary) at those layers
  - correctness of the answer (does the named species match the true species)
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import ImageFilter
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm.auto import tqdm

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

QUESTIONS = [
    "Does this image contain a cat? Answer with the name of the animal you see.",
    "Is there a cat in this picture? Tell me which animal is shown.",
    "Which animal is in this photo? Answer with its name.",
]

CAT_BREEDS = [
    "abyssinian", "bengal", "birman", "bombay", "british shorthair",
    "egyptian mau", "maine coon", "persian", "ragdoll", "russian blue",
    "siamese", "sphynx", "tabby", "calico", "tuxedo",
]
DOG_BREEDS = [
    "american bulldog", "american pit bull", "pit bull", "basset hound",
    "beagle", "boxer", "chihuahua", "cocker spaniel", "english setter",
    "german shorthaired", "great pyrenees", "havanese", "japanese chin",
    "keeshond", "leonberger", "miniature pinscher", "newfoundland",
    "pomeranian", "pug", "saint bernard", "samoyed", "scottish terrier",
    "shiba inu", "staffordshire bull terrier", "wheaten terrier",
    "yorkshire terrier", "golden retriever", "labrador", "retriever",
    "poodle", "rottweiler", "husky", "corgi", "dachshund", "dalmatian",
    "border collie", "german shepherd", "shih tzu", "schnauzer", "greyhound",
]
CAT_PATS = [r"\bcat\b", r"\bcats\b", r"\bkitten\b", r"\bkittens\b", r"\bfeline\b"] + [
    r"\b" + b.replace(" ", r"[ _-]?") + r"\b" for b in CAT_BREEDS
]
DOG_PATS = [r"\bdog\b", r"\bdogs\b", r"\bpuppy\b", r"\bpuppies\b", r"\bcanine\b"] + [
    r"\b" + b.replace(" ", r"[ _-]?") + r"\b" for b in DOG_BREEDS
]
NEG = r"(?:no|not|isn'?t|is not|aren'?t|without|doesn'?t|does not|don'?t|do not|never|nor)"


def parse_species(text: str):
    """Return 'cat', 'dog' or None for an answer string."""
    t = text.lower().replace("_", " ")
    evidence = []
    for species, pats in (("cat", CAT_PATS), ("dog", DOG_PATS)):
        for pat in pats:
            for m in re.finditer(pat, t):
                pre = t[max(0, m.start() - 30):m.start()]
                negated = bool(re.search(NEG + r"[^.;!?]{0,20}$", pre))
                evidence.append((m.start(), species, negated))
    evidence.sort()
    positive = [e for e in evidence if not e[2]]
    if positive:
        n_cat = sum(1 for e in positive if e[1] == "cat")
        n_dog = sum(1 for e in positive if e[1] == "dog")
        if n_cat == n_dog:
            return None
        return "cat" if n_cat > n_dog else "dog"
    if evidence:
        negated = {e[1] for e in evidence}
        if negated == {"cat"}:
            return "dog"
        if negated == {"dog"}:
            return "cat"
    return None


def load_model(device="cuda"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(
        MODEL, min_pixels=128 * 28 * 28, max_pixels=448 * 28 * 28
    )
    processor.tokenizer.padding_side = "left"
    return model, processor


@torch.inference_mode()
def run_batch(model, processor, images, questions, layers, max_new_tokens=16):
    messages = [
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": q},
        ]}]
        for img, q in zip(images, questions)
    ]
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages
    ]
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    captured_resp = {l: [] for l in layers}
    captured_prompt = {l: [] for l in layers}
    handles = []
    decoder_layers = model.model.language_model.layers

    def make_hook(l):
        def hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] == 1:
                captured_resp[l].append(hs[:, 0, :].detach().float().cpu())
            else:
                captured_prompt[l].append(hs[:, -1, :].detach().float().cpu())
        return hook

    for l in layers:
        handles.append(decoder_layers[l].register_forward_hook(make_hook(l)))
    try:
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    finally:
        for h in handles:
            h.remove()

    seqs = out.sequences[:, prompt_len:]
    gen_mask = inputs["attention_mask"].sum(dim=1) if "attention_mask" in inputs else None
    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    eos_ids = set(int(e) for e in eos_ids)

    n_steps = len(out.scores)
    resp_states = {l: torch.stack(captured_resp[l], dim=1) for l in layers}  # [B, T, H]
    prompt_states = {l: torch.stack(captured_prompt[l], dim=1) for l in layers}  # [B, 1, H]

    results = []
    for b in range(len(images)):
        tokens = seqs[b].tolist()
        content_len = len(tokens)
        for i, t in enumerate(tokens):
            if t in eos_ids:
                content_len = i
                break
        content_len = min(content_len, n_steps, resp_states[layers[0]].shape[1])
        content_len = max(content_len, 1 if content_len == 0 else content_len)
        logps = []
        for i in range(content_len):
            lp = torch.log_softmax(out.scores[i][b].float(), dim=-1)
            logps.append(lp[tokens[i]].item())
        jev = float(torch.exp(torch.tensor(logps).mean())) if logps else 0.0
        answer = processor.tokenizer.decode(tokens[:content_len], skip_special_tokens=True).strip()
        results.append({
            "answer": answer,
            "n_tokens": content_len,
            "token_logps": logps,
            "jev": jev,
            "resp_states": {l: resp_states[l][b, :content_len].to(torch.float16) for l in layers},
            "prompt_states": {l: prompt_states[l][b, 0].to(torch.float16) for l in layers},
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 11, 17, 23, 29, 35])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blur", type=float, default=0.0, help="max gaussian blur radius; >0 makes the task hard")
    args = ap.parse_args()

    ds = load_dataset("timm/oxford-iiit-pet")[args.split]
    n = min(args.limit, len(ds))
    order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed)).tolist()
    idxs = order[:n]
    radii = dict(zip(idxs, np.random.default_rng(args.seed + 1).uniform(1.0, args.blur, size=n)))

    model, processor = load_model()
    records = []
    n_batches = (n + args.batch - 1) // args.batch
    pbar = tqdm(total=n_batches * len(QUESTIONS), desc=args.split)
    for start in range(0, n, args.batch):
        batch_idx = idxs[start:start + args.batch]
        images = [ds[i]["image"].convert("RGB") for i in batch_idx]
        if args.blur > 0:
            images = [img.filter(ImageFilter.GaussianBlur(radius=float(radii[i]))) for img, i in zip(images, batch_idx)]
        for q_id, question in enumerate(QUESTIONS):
            outs = run_batch(model, processor, images, [question] * len(images), args.layers)
            for j, out in enumerate(outs):
                i = batch_idx[j]
                true_species = ds.features["label_cat_dog"].names[ds[i]["label_cat_dog"]]
                pred = parse_species(out["answer"])
                correct = None if pred is None else int(pred == true_species)
                rec = {
                    "image_id": ds[i]["image_id"],
                    "breed": ds.features["label"].names[ds[i]["label"]],
                    "true_species": true_species,
                    "question_id": q_id,
                    "question": question,
                    "answer": out["answer"],
                    "pred_species": pred,
                    "correct": correct,
                    "n_tokens": out["n_tokens"],
                    "token_logps": out["token_logps"],
                    "jev": out["jev"],
                    "resp_states": torch.stack([out["resp_states"][l] for l in args.layers]),
                    "prompt_states": torch.stack([out["prompt_states"][l] for l in args.layers]),
                }
                records.append(rec)
            pbar.update(1)
            pbar.set_postfix(parse_ok=round(sum(r["correct"] is not None for r in records) / len(records), 3),
                             acc=round(sum(r["correct"] or 0 for r in records) / len(records), 3))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = [{k: v for k, v in r.items() if k not in ("resp_states", "prompt_states")} for r in records]
    max_tok = max(r["n_tokens"] for r in records)
    n_layers = len(args.layers)
    d_model = records[0]["prompt_states"].shape[-1]
    resp = torch.zeros(len(records), n_layers, max_tok, d_model, dtype=torch.float16)
    for i, r in enumerate(records):
        t = r["resp_states"].shape[1]
        resp[i, :, :t] = r["resp_states"]
    payload = {
        "questions": QUESTIONS,
        "layers": args.layers,
        "split": args.split,
        "meta": meta,
        "resp_states": resp,
        "prompt_states": torch.stack([r["prompt_states"] for r in records]),
    }
    torch.save(payload, args.out)
    print(f"saved {len(records)} records -> {args.out}")
    for r in records[:5]:
        print(f"  [{r['true_species']}] q{r['question_id']} '{r['answer'][:60]}' -> {r['pred_species']} "
              f"correct={r['correct']} jev={r['jev']:.3f} ntok={r['n_tokens']}")


if __name__ == "__main__":
    main()
