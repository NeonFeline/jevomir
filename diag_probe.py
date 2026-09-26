"""Probe output distribution per subset x split x answer kind."""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from train_probe import CalibProbe
from train_probe_cauldron import load_shards, split_of

meta, x, layers = load_shards(Path(sys.argv[1]))
ck = torch.load(Path(sys.argv[2]) / "probe.pt", weights_only=False, map_location="cpu")
probe = CalibProbe(ck["n_layers"], ck["d_model"], d_proj=ck["d_proj"])
probe.load_state_dict(ck["state_dict"]); probe.eval()
with torch.no_grad():
    p = torch.cat([torch.sigmoid(probe((x[i:i + 8192] - ck["mu"]) / ck["sd"])) for i in range(0, len(x), 8192)]).numpy()
y = np.array([m["correct"] for m in meta]); raw = np.array([m["conf"] for m in meta])
groups = defaultdict(list)
for i, m in enumerate(meta):
    groups[(m["subset"], split_of(m["image_key"], 0.15, 0.1), m["answer_kind"])].append(i)
print(f"{'subset':9s} {'split':5s} {'kind':11s} {'n':>6s} {'acc':>5s} {'probe mean':>10s} {'frac<0.01':>9s} {'frac>0.99':>9s} {'AUROC probe':>11s} {'raw':>5s}")
for key in sorted(groups):
    idx = np.array(groups[key])
    if len(idx) < 50:
        continue
    auc = lambda s: roc_auc_score(y[idx], s[idx]) if len(set(y[idx])) > 1 else float("nan")
    print(f"{key[0]:9s} {key[1]:5s} {key[2]:11s} {len(idx):6d} {y[idx].mean():5.3f} {p[idx].mean():10.3f} "
          f"{(p[idx] < 0.01).mean():9.3f} {(p[idx] > 0.99).mean():9.3f} {auc(p):11.3f} {auc(raw):5.3f}")
