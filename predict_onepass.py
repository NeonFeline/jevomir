"""One-pass deployment: image -> single forward pass -> P(cat)/P(dog) + calibrated P(answer correct)."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from extract_onepass import MODEL, PROTOCOLS, cand_ids, forward_batch, load_model
from train_probe import CalibProbe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--protocol", type=int, default=0, choices=[p["id"] for p in PROTOCOLS])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--probe", type=Path, default=Path("artifacts/q35/probe.pt"))
    ap.add_argument("--features", type=Path, default=Path("artifacts/q35/features_train.pt"))
    ap.add_argument("--blur", type=float, default=0.0)
    args = ap.parse_args()

    layers = torch.load(args.features, weights_only=False, mmap=True)["layers"]
    ckpt = torch.load(args.probe, map_location="cpu", weights_only=False)
    d_proj = ckpt["state_dict"]["proj.0.weight"].shape[0]
    probe = CalibProbe(ckpt["n_layers"], ckpt["d_model"], d_proj=d_proj)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    model, processor = load_model(args.model)
    proto = PROTOCOLS[args.protocol]
    ids = {w: cand_ids(processor.tokenizer, w) for w in ("cat", "dog", "yes", "no")}
    images = [Image.open(p).convert("RGB") for p in args.images]
    if args.blur > 0:
        from PIL import ImageFilter
        images = [img.filter(ImageFilter.GaussianBlur(radius=args.blur)) for img in images]

    p_vals, states = forward_batch(model, processor, images, proto["question"], layers,
                                   ids[proto["pos"]], ids[proto["neg"]])
    x = states.float()
    x = (x - ckpt["mu"]) / ckpt["sd"]
    with torch.no_grad():
        cal = torch.sigmoid(probe(x)).tolist()

    for path, p_cat, c in zip(args.images, p_vals, cal):
        verdict_cat = p_cat >= 0.5
        conf = p_cat if verdict_cat else 1 - p_cat
        print(path)
        print(f"  Q: {proto['question']}")
        print(f"  A: {'cat' if verdict_cat else 'dog'}")
        print(f"  one-pass raw confidence (max softmax over candidates): {conf:.3f}")
        print(f"  calibrated P(answer correct)                          : {c:.3f}")


if __name__ == "__main__":
    main()
