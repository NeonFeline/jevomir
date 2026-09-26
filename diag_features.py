"""Why does the probe collapse on some subsets? Inspect raw and standardized features."""
import sys
from pathlib import Path
import torch
from train_probe_cauldron import load_shards, split_of

meta, x, layers = load_shards(Path(sys.argv[1]))
subs = [m["subset"] for m in meta]
print("non-finite values:", (~torch.isfinite(x)).sum().item(), "| fp16 max-ish |x| >= 60000:", (x.abs() >= 60000).sum().item())
split = [split_of(m["image_key"], 0.15, 0.1) for m in meta]
tr = torch.tensor([s == "train" and sub != "vsr" for s, sub in zip(split, subs)])
mu, sd = x[tr].mean(0, keepdim=True), x[tr].std(0, keepdim=True)
print("train sd < 1e-3 per layer:", [(sd[0, i] < 1e-3).sum().item() for i in range(len(layers))])
z = (x - mu) / sd.clamp(min=1e-6)
for s in sorted(set(subs)):
    idx = torch.tensor([u == s for u in subs])
    zmax = z[idx].abs().amax(dim=(1, 2))
    raw = x[idx].abs().amax(dim=(1, 2))
    print(f"{s:9s} n={int(idx.sum()):6d} |z| max median {zmax.median():9.1f} p99 {zmax.quantile(0.99):10.1f} | raw |x| max median {raw.median():8.1f}")
# Which dims blow up on vsr/vqav2?
for s in ("vsr", "vqav2", "clevr"):
    idx = torch.tensor([u == s for u in subs])
    zz = z[idx].abs().amax(0)  # [L, D]
    top = torch.topk(zz.flatten(), 5)
    print(s, "worst (layer, dim, |z|, train sd, train mu, subset mean):",
          [(layers[i // zz.shape[1]], i % zz.shape[1], round(v.item(), 1), round(sd.flatten()[i].item(), 5),
            round(mu.flatten()[i].item(), 3), round(x[idx].reshape(int(idx.sum()), -1)[:, i].mean().item(), 3))
           for v, i in zip(top.values, top.indices)])
