"""Paired held-out comparison of answer models (base / LoRA adapters) on the same items.

Every model sees the identical test-bucket items (perceptual-image-hash split: no image is
shared with any training stage), once in the original option order and once permuted (never
the identity). Per item we keep each model's option probabilities, so the comparison is
paired: accuracy / NLL / Brier / ECE per subset with paired-bootstrap CIs for the
differences, McNemar flips (right->wrong vs wrong->right), and order robustness.

    torchrun --nproc_per_node 4 eval_compare.py --out runs/X/compare \
        --model-spec base=- rft=runs/X/rft/adapter-round1 new=runs/X/grpo-head/adapter \
        --eval-subsets tallyqa:1500,nlvr2:1500,iconqa:1500,clevr:1500,vqav2:1500

Writes items.jsonl (one row per item, every model's probabilities and picks), report.json
and REPORT.md. The first --model-spec with a path is the paired-test baseline (--baseline).
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForImageTextToText

import dist_utils as du
from cauldron_tasks import LETTERS
from train_grpo import GRPOCollate, letter_logps
from train_probe import ece
from train_qlora import MODEL_ID, MODEL_REVISION, ItemDataset, load_pool, load_processor, parse_sizes


def parse_specs(specs):
    out = []
    for s in specs:
        name, _, path = s.partition("=")
        if not name or not path:
            raise SystemExit(f"--model-spec wants name=path (or name=- for the base model), got {s!r}")
        out.append((name, None if path == "-" else Path(path)))
    return out


def load_models(args, specs):
    """One base model; every adapter loaded into it under its own name (switched per pass)."""
    device = torch.cuda.current_device()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": device})
    adapters = [(n, p) for n, p in specs if p is not None]
    if adapters:
        model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0])
        for n, p in adapters[1:]:
            model.load_adapter(p, adapter_name=n)
    return model.eval()


@torch.inference_mode()
def score(model, specs, pool, processor, letters, args):
    """Per item: every model's probabilities in original option order, from both prompt orders."""
    rows = {}
    mine = du.shard(list(range(len(pool))))
    if not mine:
        return rows
    loader = DataLoader(ItemDataset([pool[i] for i in mine]), batch_size=args.batch, num_workers=args.workers,
                        collate_fn=GRPOCollate(processor, args.image_size, args.pad_multiple, "eval"),
                        pin_memory=True)
    for b, (batch, inputs, perm_inputs, orders) in enumerate(loader):
        for name, path in specs:
            if path is None:
                ctx = model.disable_adapter() if isinstance(model, PeftModel) else _null()
            else:
                model.set_adapter(name)
                ctx = _null()
            with ctx:
                lps = letter_logps(model, inputs, batch, letters)
                lps_perm = letter_logps(model, perm_inputs, batch, letters)
            for it, lp, lpp, order in zip(batch, lps, lps_perm, orders):
                inv = torch.argsort(torch.tensor(order))
                i = mine[it["index"]]
                r = rows.setdefault(i, {"models": {}})
                r["models"][name] = {"p": lp.exp().cpu().tolist(),
                                     "p_perm": lpp.exp().cpu()[inv].tolist()}
        if b % 20 == 0:
            du.print0(f"  batch {b}/{len(loader)}")
    for i, r in rows.items():
        it = pool[i]
        r.update({"subset": it["subset"], "image_id": it["image_id"], "qa_index": it.get("qa_index", 0),
                  "question": it["question"], "options": it["options"], "answer": int(it["answer"]),
                  "answer_kind": it["answer_kind"]})
    return rows


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def metrics(P, y, Pp):
    """P, Pp: lists of prob vectors (original / permuted prompt), y: answer indices."""
    pick = np.array([int(np.argmax(p)) for p in P])
    pick_perm = np.array([int(np.argmax(p)) for p in Pp])
    conf = np.array([float(np.max(p)) for p in P])
    p_ans = np.array([float(p[a]) for p, a in zip(P, y)])
    correct = (pick == y).astype(float)
    brier = np.array([float(sum((q - (j == a)) ** 2 for j, q in enumerate(p))) for p, a in zip(P, y)])
    return {"correct": correct, "conf": conf, "nll": -np.log(np.clip(p_ans, 1e-12, 1)), "brier": brier,
            "correct_perm": (pick_perm == y).astype(float), "agree": (pick == pick_perm).astype(float),
            "pick": pick}


def summary(m):
    c = np.clip(m["conf"], 1e-6, 1 - 1e-6)
    return {"n": int(len(m["correct"])), "acc": float(m["correct"].mean()), "nll": float(m["nll"].mean()),
            "brier": float(m["brier"].mean()), "ece": float(ece(c, m["correct"])[0]),
            "mean_conf": float(m["conf"].mean()), "acc_permuted": float(m["correct_perm"].mean()),
            "order_agreement": float(m["agree"].mean())}


def mcnemar_p(b, c):
    """Exact two-sided McNemar p-value: b = base right/new wrong, c = base wrong/new right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def paired(mb, mn, n_boot, rng):
    """New minus baseline, with paired-bootstrap 95% CIs (items resampled jointly)."""
    n = len(mb["correct"])
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for key in ("acc", "nll", "brier", "ece"):
        def stat(m, ix):
            if key == "acc":
                return m["correct"][ix].mean(axis=-1)
            if key in ("nll", "brier"):
                return m[key][ix].mean(axis=-1)
            c = np.clip(m["conf"], 1e-6, 1 - 1e-6)
            return np.array([ece(c[r], m["correct"][r])[0] for r in ix])
        full = np.arange(n)[None]
        point = float((stat(mn, full) - stat(mb, full))[0])
        boots = stat(mn, idx) - stat(mb, idx)
        out[key] = {"delta": point, "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5))}
    b = int(((mb["correct"] == 1) & (mn["correct"] == 0)).sum())
    c = int(((mb["correct"] == 0) & (mn["correct"] == 1)).sum())
    out["flips"] = {"right_to_wrong": b, "wrong_to_right": c, "mcnemar_p": mcnemar_p(b, c),
                    "answer_changed": int((mb["pick"] != mn["pick"]).sum())}
    return out


def fmt_ci(d, digits=3):
    f = f"{{:+.{digits}f}}"
    star = " *" if d["lo"] > 0 or d["hi"] < 0 else ""
    return f"{f.format(d['delta'])} [{f.format(d['lo'])}, {f.format(d['hi'])}]{star}"


def report(rows, specs, args):
    names = [n for n, _ in specs]
    baseline = args.baseline or next(n for n, p in specs if p is not None)
    others = [n for n in names if n != baseline]
    rng = np.random.default_rng(args.seed)
    subsets = sorted({r["subset"] for r in rows}) + ["ALL"]
    res = {"models": {}, "paired": {}, "baseline": baseline}
    per = {}
    for s in subsets:
        sel = [r for r in rows if s == "ALL" or r["subset"] == s]
        y = np.array([r["answer"] for r in sel])
        per[s] = {n: metrics([r["models"][n]["p"] for r in sel], y, [r["models"][n]["p_perm"] for r in sel])
                  for n in names}
        res["models"][s] = {n: summary(per[s][n]) for n in names}
        res["paired"][s] = {o: paired(per[s][baseline], per[s][o], args.boot, rng) for o in others}

    L = [f"# Held-out comparison ({len(rows)} test-bucket items, images disjoint from all training)", "",
         "Models: " + ", ".join(f"`{n}` = {p or 'base model, no adapter'}" for n, p in specs), "",
         "## Per model", "",
         "| subset | model | n | acc | acc (permuted opts) | order agreement | NLL | Brier | ECE | mean conf |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in subsets:
        for n in names:
            d = res["models"][s][n]
            L.append(f"| {s} | {n} | {d['n']} | {d['acc']:.3f} | {d['acc_permuted']:.3f} | "
                     f"{d['order_agreement']:.3f} | {d['nll']:.3f} | {d['brier']:.3f} | {d['ece']:.4f} | "
                     f"{d['mean_conf']:.3f} |")
    for o in others:
        L += ["", f"## `{o}` minus `{baseline}` (paired bootstrap 95% CI, {args.boot} resamples; "
                  "* = CI excludes 0)", "",
              "Lower is better for NLL, Brier and ECE.", "",
              "| subset | Δacc | ΔNLL | ΔBrier | ΔECE | right→wrong | wrong→right | McNemar p | answers changed |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in subsets:
            d = res["paired"][s][o]
            f = d["flips"]
            L.append(f"| {s} | {fmt_ci(d['acc'])} | {fmt_ci(d['nll'])} | {fmt_ci(d['brier'])} | "
                     f"{fmt_ci(d['ece'], 4)} | {f['right_to_wrong']} | {f['wrong_to_right']} | "
                     f"{f['mcnemar_p']:.3g} | {f['answer_changed']} |")
        changed = [r for r in rows if int(np.argmax(r["models"][o]["p"])) != int(np.argmax(r["models"][baseline]["p"]))]
        rng2 = np.random.default_rng(args.seed + 1)
        L += ["", f"### Sample of changed answers (`{baseline}` → `{o}`)", "",
              "| subset | question | options | gold | " + f"{baseline} | {o} |", "|---|---|---|---|---|---|"]
        for r in [changed[i] for i in rng2.permutation(len(changed))[:args.examples]]:
            def cell(n):
                p = r["models"][n]["p"]
                k = int(np.argmax(p))
                mark = "✓" if k == r["answer"] else "✗"
                return f"{LETTERS[k]} ({p[k]:.2f}) {mark}"
            q = r["question"].replace("|", "/").replace("\n", " ")[:90]
            opts = "; ".join(f"{LETTERS[j]}={str(t)[:18]}" for j, t in enumerate(r["options"])).replace("|", "/")
            L.append(f"| {r['subset']} | {q} | {opts} | {LETTERS[r['answer']]} | {cell(baseline)} | {cell(o)} |")
    return res, "\n".join(L) + "\n"


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-spec", nargs="+", required=True, help="name=adapter_path, or name=- for the base model")
    ap.add_argument("--baseline", default="", help="model name the others are compared against")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("JEV_DATA", Path.home() / "cauldron")))
    ap.add_argument("--eval-subsets", default="tallyqa:1500,nlvr2:1500,iconqa:1500,clevr:1500,vqav2:1500")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--examples", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    specs = parse_specs(args.model_spec)

    du.init()
    try:
        if du.is_main():
            if (args.out / "report.json").exists():
                raise SystemExit(f"{args.out}/report.json exists; pick a new --out")
            args.out.mkdir(parents=True, exist_ok=True)
        du.barrier()
        pool = load_pool(args.data, parse_sizes(args.eval_subsets), "test", args.seed)
        processor, letters = load_processor(args)
        model = load_models(args, specs)
        du.print0(f"scoring {len(pool)} items x {len(specs)} models x 2 option orders ...")
        rows = {}
        for part in du.gather(score(model, specs, pool, processor, letters, args)):
            rows.update(part)
        if du.is_main():
            rows = [rows[i] for i in sorted(rows)]
            with open(args.out / "items.jsonl", "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            res, md = report(rows, specs, args)
            (args.out / "report.json").write_text(json.dumps(res, indent=2))
            (args.out / "REPORT.md").write_text(md)
            print(md, flush=True)
        du.barrier()
    finally:
        du.cleanup()


if __name__ == "__main__":
    main()
