"""End-to-end inference: image + question -> VLM answer + raw jev prob + calibrated probe prob."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageFilter

from extract import QUESTIONS, load_model, run_batch
from train_probe import CalibProbe


def predict(images, question, model, processor, ckpt, layers):
    outs = run_batch(model, processor, images, [question] * len(images), layers)
    ntok = torch.tensor([o["n_tokens"] for o in outs])
    T, d_model = int(ntok.max()), ckpt["d_model"]
    resp = torch.zeros(len(outs), len(layers), T, d_model)
    for b, o in enumerate(outs):
        for li, l in enumerate(layers):
            resp[b, li, :o["n_tokens"]] = o["resp_states"][l]
    prompt = torch.stack([torch.stack([o["prompt_states"][l] for l in layers]) for o in outs])  # [B,L,D]
    mask = torch.arange(T)[None, :] < ntok[:, None]
    pooled = (resp * mask[:, None, :, None]).sum(2) / ntok[:, None, None].clamp(min=1)
    x = torch.cat([pooled, prompt.float()], dim=1)
    mu, sd = ckpt["mu"], ckpt["sd"]
    x = (x - mu) / sd
    d_proj = ckpt["state_dict"]["proj.0.weight"].shape[0]
    probe = CalibProbe(ckpt["n_layers"], ckpt["d_model"], d_proj=d_proj)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()
    with torch.no_grad():
        probs = torch.sigmoid(probe(x.to(next(probe.parameters()).device))).cpu()
    return outs, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--question", default=QUESTIONS[0])
    ap.add_argument("--probe", type=Path, default=Path("artifacts/probe.pt"))
    ap.add_argument("--features", type=Path, default=Path("artifacts/features_train.pt"))
    ap.add_argument("--blur", type=float, default=0.0)
    args = ap.parse_args()

    layers = torch.load(args.features, weights_only=False, mmap=True)["layers"]
    ckpt = torch.load(args.probe, map_location="cpu", weights_only=False)
    model, processor = load_model()
    images = [Image.open(p).convert("RGB") for p in args.images]
    if args.blur > 0:
        images = [img.filter(ImageFilter.GaussianBlur(radius=args.blur)) for img in images]
    outs, probs = predict(images, args.question, model, processor, ckpt, layers)
    for path, o, p in zip(args.images, outs, probs):
        print(f"{path}")
        print(f"  Q: {args.question}")
        print(f"  A: {o['answer']!r}")
        print(f"  raw jev prob (exp(mean log p_i)) : {o['jev']:.3f}   ({o['n_tokens']} tokens)")
        print(f"  calibrated P(answer correct)     : {p.item():.3f}")


if __name__ == "__main__":
    main()
