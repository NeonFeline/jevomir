"""Train a small multi-layer calibration probe on Qwen2.5-VL hidden states.

Target: P(answer is correct). Input: pooled hidden states of the generated answer
tokens + last-prompt-token states, from 6 LLM layers each.

Evaluates ECE / Brier / AUROC of the raw "jev" probability (geometric mean token
prob) vs the probe, on a held-out image split, and draws reliability diagrams.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class CalibProbe(nn.Module):
    def __init__(self, n_layers, d_model, d_proj=64, hidden=256, dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.ModuleList([nn.Linear(d_model, d_proj) for _ in range(n_layers)])
        self.mlp = nn.Sequential(
            nn.Linear(n_layers * d_proj, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):  # [B, L, D]
        x = self.ln(x)
        z = torch.cat([p(x[:, i]) for i, p in enumerate(self.proj)], dim=-1)
        return self.mlp(z).squeeze(-1)


def load_features(path):
    d = torch.load(path, weights_only=False)
    meta, prompt = d["meta"], d["prompt_states"]
    if "resp_states" in d:
        resp = d["resp_states"]
        ntok = torch.tensor([m["n_tokens"] for m in meta])
        T = resp.shape[2]
        mask = (torch.arange(T)[None, :] < ntok[:, None]).float()
        pooled = (resp.float() * mask[:, None, :, None]).sum(2) / ntok[:, None, None].clamp(min=1)
        x = torch.cat([pooled, prompt.float()], dim=1)  # [N, 2L, D]
        raw_name = "raw jev"
    else:
        x = prompt.float()  # [N, L, D]
        raw_name = "raw conf"
    y = torch.tensor([m["correct"] if m["correct"] is not None else -1 for m in meta])
    raw = torch.tensor([m.get("jev", m.get("conf")) for m in meta])
    return meta, x, y, raw, raw_name


def ece(probs, labels, n_bins=15):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(probs, bins[1:-1])
    e, rows = 0.0, []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append({"lo": float(bins[b]), "hi": float(bins[b + 1]), "n": 0})
            continue
        conf, acc = float(probs[m].mean()), float(labels[m].mean())
        e += float(m.mean()) * abs(acc - conf)
        rows.append({"lo": float(bins[b]), "hi": float(bins[b + 1]), "n": int(m.sum()),
                     "conf": conf, "acc": acc})
    return e, rows


def report(name, probs, labels):
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    e, rows = ece(p, labels)
    out = {
        "ece": e,
        "brier": float(np.mean((p - labels) ** 2)),
        "auroc": float(roc_auc_score(labels, p)) if len(np.unique(labels)) > 1 else None,
        "mean_conf": float(p.mean()),
        "accuracy": float(labels.mean()),
        "bins": rows,
    }
    auroc = "n/a" if out["auroc"] is None else f"{out['auroc']:.4f}"
    print(f"[{name}] ECE={out['ece']:.4f} Brier={out['brier']:.4f} AUROC={auroc} "
          f"mean_conf={out['mean_conf']:.4f} acc={out['accuracy']:.4f}")
    return out


def reliability_plot(raw, probe, labels, path, raw_title="raw jev probability = exp(mean log p_i)"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, probs, title in [(axes[0], raw, raw_title),
                             (axes[1], probe, "calibration probe (multi-layer hidden states)")]:
        e, rows = ece(probs, labels)
        centers = [(r["lo"] + r["hi"]) / 2 for r in rows if r["n"] > 0]
        accs = [r["acc"] for r in rows if r["n"] > 0]
        confs = [r["conf"] for r in rows if r["n"] > 0]
        ns = [r["n"] for r in rows if r["n"] > 0]
        ax.bar(centers, accs, width=0.06, alpha=0.5, color="tab:blue", label="empirical accuracy")
        ax.plot(centers, confs, "o-", color="tab:red", ms=4, label="stated confidence")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        ax.set_title(f"{title}\nECE={e:.3f}")
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for c, a, n in zip(centers, accs, ns):
            ax.text(c, a + 0.03, str(n), ha="center", fontsize=5, color="tab:blue")
    axes[0].set_ylabel("accuracy / confidence")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=Path("artifacts/features_train.pt"))
    ap.add_argument("--test", type=Path, default=Path("artifacts/features_test.pt"))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-proj", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    meta_tr, x_tr, y_tr, jev_tr, raw_name = load_features(args.train)
    meta_te, x_te, y_te, jev_te, _ = load_features(args.test)

    ok_tr = y_tr >= 0
    ok_te = y_te >= 0
    print(f"train pairs {len(y_tr)} (parse_ok {ok_tr.float().mean():.3f}), "
          f"test pairs {len(y_te)} (parse_ok {ok_te.float().mean():.3f})")

    # val split by image (no image leaks between train/val)
    image_ids = np.array([m["image_id"] for m in meta_tr])
    uniq = np.unique(image_ids)
    rng = np.random.default_rng(args.seed)
    val_imgs = set(rng.choice(uniq, size=max(1, int(len(uniq) * args.val_frac)), replace=False).tolist())
    val_mask = np.array([i in val_imgs for i in image_ids]) & ok_tr.numpy()
    tr_mask = ~val_mask & ok_tr.numpy()
    print(f"images: {len(uniq)} train, {len(val_imgs)} val; pairs: {tr_mask.sum()} train, {val_mask.sum()} val")

    # standardize with train stats (features stored as fp16)
    mu = x_tr[tr_mask].mean(dim=0, keepdim=True)
    sd = x_tr[tr_mask].std(dim=0, keepdim=True).clamp(min=1e-6)
    xs = (x_tr - mu) / sd
    x_te_s = (x_te - mu) / sd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_tr_d, y_tr_d = xs[tr_mask].to(device), y_tr[tr_mask].float().to(device)
    x_val_d, y_val_d = xs[val_mask].to(device), y_tr[val_mask].float().to(device)
    x_te_d = x_te_s.to(device)

    n_layers, d_model = x_tr.shape[1], x_tr.shape[2]
    model = CalibProbe(n_layers, d_model, d_proj=args.d_proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    n = int(tr_mask.sum())
    best_val, best_state, bad = float("inf"), None, 0

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = lossf(model(x_tr_d[idx]), y_tr_d[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = lossf(model(x_val_d), y_val_d).item()
        if val_loss < best_val - 1e-5:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}, best val loss {best_val:.4f}")
                break
        if epoch % 20 == 0:
            print(f"epoch {epoch}: train loss {tot / n:.4f} val loss {val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probe_te = torch.sigmoid(model(x_te_d)).cpu().numpy()

    labels = y_te[ok_te].numpy()
    raw = jev_te[ok_te].numpy()
    probe = probe_te[ok_te.numpy()]

    results = {
        "n_train_pairs": int(ok_tr.sum()),
        "n_test_pairs": int(ok_te.sum()),
        "n_train_images": int(len(uniq) - len(val_imgs)),
        "n_val_images": int(len(val_imgs)),
        "probe_arch": f"{n_layers}x Linear({d_model}->{args.d_proj}) -> MLP({n_layers * args.d_proj}-256-128-1)",
        "accuracy": float(labels.mean()),
        "parse_ok_rate_test": float(ok_te.float().mean()),
        raw_name: report(raw_name, raw, labels),
        "probe": report("probe", probe, labels),
    }

    # per-question metrics
    per_q = {}
    qkey = "question_id" if "question_id" in meta_te[0] else "protocol_id"
    q_all = np.array([m[qkey] for m in meta_te])[ok_te.numpy()]
    q_texts = {}
    for m in meta_te:
        q_texts[m[qkey]] = m["question"]
    for q in sorted(set(q_all.tolist())):
        m = q_all == q
        per_q[str(q)] = {
            "question": q_texts[q],
            "n": int(m.sum()),
            "accuracy": float(labels[m].mean()),
            raw_name: report(f"raw q{q}", raw[m], labels[m]),
            "probe": report(f"probe q{q}", probe[m], labels[m]),
        }
    results["per_question"] = per_q

    # what did the probe do to the mid-confidence cases?
    story = {}
    for name, p in (("raw", raw), ("probe", probe)):
        mid = (p > 0.4) & (p < 0.6)
        if mid.sum() >= 10:
            story[name] = {"n": int(mid.sum()), "mean_prob": float(p[mid].mean()),
                           "empirical_accuracy": float(labels[mid].mean())}
    results["mid_confidence_0.4_0.6"] = story

    # examples: confident-wrong and mid-confidence
    examples = []
    ok_idx = np.where(ok_te.numpy())[0]
    for j, i in enumerate(ok_idx):
        if 0.3 < raw[j] < 0.7 or (labels[j] == 0 and probe[j] > 0.5):
            m = meta_te[i]
            examples.append({
                "image_id": m["image_id"], "breed": m.get("breed"), "true_species": m.get("true_species"),
                "question": m["question"][:60], "answer": m["answer"][:80], "correct": int(m["correct"]),
                "raw_jev": float(raw[j]), "probe": float(probe[j]), "n_tokens": m.get("n_tokens", 0),
            })
    examples.sort(key=lambda e: e["raw_jev"])
    results["examples"] = examples[:40]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    torch.save({"state_dict": best_state, "n_layers": n_layers, "d_model": d_model,
                "d_proj": args.d_proj, "mu": mu, "sd": sd}, args.out / "probe.pt")
    style = ("raw jev probability = exp(mean log p_i)" if raw_name == "raw jev"
             else "raw confidence = max(P(cat), P(dog)) from a single forward pass")
    reliability_plot(raw, probe, labels, args.out / "reliability.png", raw_title=style)

    print("\n--- mid-confidence story (0.4 < p < 0.6) ---")
    for k, v in story.items():
        print(f"  {k:5s}: n={v['n']:4d} stated={v['mean_prob']:.3f} actual={v['empirical_accuracy']:.3f}")


if __name__ == "__main__":
    main()
