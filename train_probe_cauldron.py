"""Train the calibration probe on Cauldron shards.

Splits by perceptual image key (shared COCO photos never straddle train/test), and keeps
whole subsets out of training to test transfer to an unseen data type.
Reports raw max-letter-probability vs probe per subset: ECE, Brier, AUROC.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from train_probe import CalibProbe, reliability_plot, report


def load_shards(directory: Path):
    metas, states, layers = [], [], None
    for path in sorted(directory.glob("*.pt")):
        d = torch.load(path, weights_only=False)
        if layers is not None and d["layers"] != layers:
            raise ValueError(f"{path} uses layers {d['layers']}, expected {layers}")
        layers = d["layers"]
        metas += d["meta"]
        states.append(d["prompt_states"])
    return metas, torch.cat(states).float(), layers


def split_of(key: str, test_frac: float, val_frac: float) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "val" if h < test_frac + val_frac else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True, help="directory of per-subset shards")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--holdout-subsets", nargs="*", default=["vsr"])
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-proj", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)

    meta, x, layers = load_shards(args.features)
    y = torch.tensor([m["correct"] for m in meta]).float()
    raw = np.array([m["conf"] for m in meta])
    subsets = np.array([m["subset"] for m in meta])
    held = np.isin(subsets, args.holdout_subsets)
    split = np.array([split_of(m["image_key"], args.test_frac, args.val_frac) for m in meta])
    tr, va = (split == "train") & ~held, (split == "val") & ~held
    te_in, te_out = (split == "test") & ~held, held
    # Remove held-out images from in-distribution training even if another subset reuses them.
    held_keys = {m["image_key"] for m, h in zip(meta, held) if h}
    leak = np.array([m["image_key"] in held_keys for m in meta]) & ~held
    tr, va = tr & ~leak, va & ~leak
    print(f"items {len(meta)} | train {tr.sum()} val {va.sum()} test-in {te_in.sum()} "
          f"test-holdout {te_out.sum()} | dropped for holdout image overlap {leak.sum()}")

    mu = x[tr].mean(0, keepdim=True)
    sd = x[tr].std(0, keepdim=True).clamp(min=1e-6)
    xs = (x - mu) / sd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, ytr = xs[tr].to(device), y[tr].to(device)
    xva, yva = xs[va].to(device), y[va].to(device)

    model = CalibProbe(x.shape[1], x.shape[2], d_proj=args.d_proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(ytr), device=device)
        for i in range(0, len(ytr), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            lossf(model(xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = lossf(model(xva), yva).item()
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    print(f"stopped at epoch {epoch}, best val loss {best:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probe = torch.sigmoid(model(xs.to(device))).cpu().numpy()

    labels = y.numpy()
    results = {"layers": layers, "holdout_subsets": args.holdout_subsets,
               "n": {"train": int(tr.sum()), "val": int(va.sum()), "test_in": int(te_in.sum()),
                     "test_holdout": int(te_out.sum())}}
    for name, mask in (("test_in", te_in), ("test_holdout", te_out)):
        if mask.sum() == 0:
            continue
        block = {"raw": report(f"{name} raw", raw[mask], labels[mask]),
                 "probe": report(f"{name} probe", probe[mask], labels[mask]), "per_subset": {}}
        for s in sorted(set(subsets[mask])):
            m = mask & (subsets == s)
            block["per_subset"][s] = {"n": int(m.sum()), "accuracy": float(labels[m].mean()),
                                      "raw": report(f"  {s} raw", raw[m], labels[m]),
                                      "probe": report(f"  {s} probe", probe[m], labels[m])}
        results[name] = block
        reliability_plot(raw[mask], probe[mask], labels[mask], args.out / f"reliability_{name}.png",
                         raw_title="raw = max option-letter probability")
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    torch.save({"state_dict": best_state, "n_layers": x.shape[1], "d_model": x.shape[2],
                "d_proj": args.d_proj, "mu": mu, "sd": sd, "layers": layers}, args.out / "probe.pt")


if __name__ == "__main__":
    main()
