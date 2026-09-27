"""Final step of the SFT -> RFT -> GRPO pipeline: merge, re-evaluate, report.

Merges the last adapter into the bf16 base model (a standalone checkpoint that loads with
AutoModelForImageTextToText, no PEFT needed), evaluates the merged model on the same held-out
test pool the stages used (option-letter accuracy / confidence / ECE, plus the verbalized
confidence if GRPO trained it) and writes REPORT.md + report.json next to it.

    torchrun --nproc_per_node 8 finalize_pipeline.py --runs /raid/.../runs/pipe-001 \
        --adapter /raid/.../runs/pipe-001/grpo/adapter --out /raid/.../runs/pipe-001/final
"""

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

import dist_utils as du
from train_grpo import confidence_ids, eval_confidence
from train_qlora import (MODEL_ID, MODEL_REVISION, eval_all, load_pool, load_processor, parse_sizes,
                         seed_everything)


def load_metrics(path):
    p = Path(path) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def fmt_row(name, ev, conf=None):
    cells = []
    for s in sorted(ev):
        d = ev[s]
        cells.append(f"{s}: acc {d['acc']:.3f} / ece {d['ece']:.3f}")
    extra = ""
    if conf:
        extra = " | verbalized: " + ", ".join(f"{s} ece {d['conf_ece']:.3f} (conf {d['mean_conf']:.2f}, auroc {d.get('conf_auroc', float('nan')):.2f})"
                                                for s, d in sorted(conf.items()))
    return f"| {name} | " + "; ".join(cells) + extra + " |"


def macro(ev, key):
    return sum(d[key] for d in ev.values()) / max(1, len(ev))


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True, help="pipeline dir with sft/ rft/ grpo/")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("JEV_DATA", Path.home() / "cauldron")))
    ap.add_argument("--eval-subsets", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=MODEL_REVISION)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--pad-multiple", type=int, default=128)
    ap.add_argument("--eval-batch", type=int, default=16)
    ap.add_argument("--eval-workers", type=int, default=4)
    ap.add_argument("--confidence", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    du.init()
    try:
        run(args)
    finally:
        du.cleanup()


def run(args):
    seed_everything(args.seed)
    if du.is_main():
        args.out.mkdir(parents=True, exist_ok=True)
    du.barrier()
    eval_pool = load_pool(args.data, parse_sizes(args.eval_subsets), "test", args.seed)
    processor, letters = load_processor(args)

    base = AutoModelForImageTextToText.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()})
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload().eval()
    if du.is_main():
        model.save_pretrained(args.out / "model", safe_serialization=True)
        processor.save_pretrained(args.out / "model")
    du.barrier()
    du.print0("merged model ->", args.out / "model")

    final = eval_all(model, eval_pool, args, processor, letters)
    final_conf = {}
    if args.confidence:
        conf_ids, conf_vals = confidence_ids(processor.tokenizer)
        final_conf = eval_confidence(model, eval_pool, args, processor, letters, conf_ids, conf_vals)
    if not du.is_main():
        return

    sft, rft, grpo = (load_metrics(args.runs / s) for s in ("sft", "rft", "grpo"))
    rows, stages = [], {}
    if sft:
        stages["base"] = sft["before"]
        rows.append(fmt_row("base Qwen3.5-4B", sft["before"]))
        stages["sft"] = sft["after"]
        rows.append(fmt_row(f"SFT ({sft.get('steps')}/{sft.get('planned_steps')} steps)", sft["after"]))
    if rft:
        for r in rft["rounds"]:
            stages[f"rft{r['round']}"] = r["eval"]
            rows.append(fmt_row(f"RFT round {r['round']} (kept {r['kept']})", r["eval"]))
    if grpo:
        stages["grpo"] = grpo["after"]
        rows.append(fmt_row(f"GRPO ({grpo.get('steps')}/{grpo.get('planned_steps')} steps)", grpo["after"],
                            grpo.get("after_conf")))
    rows.append(fmt_row("final merged model", final, final_conf))

    summary = {name: {"macro_acc": macro(ev, "acc"), "macro_ece": macro(ev, "ece")}
               for name, ev in {**stages, "final": final}.items() if ev}
    report = {"adapter": str(args.adapter), "model": str(args.out / "model"), "summary": summary,
              "final": final, "final_conf": final_conf, "stages": stages,
              "grpo_before_conf": (grpo or {}).get("before_conf", {})}
    (args.out / "report.json").write_text(json.dumps(report, indent=2))

    lines = ["# jevomir pipeline report", "",
             f"Final model (merged, bf16): `{args.out / 'model'}`  ",
             f"Final adapter: `{args.adapter}`  ",
             f"Held-out test pool: {len(eval_pool)} items ({args.eval_subsets}), split by perceptual image hash.",
             "", "## Macro average over subsets", "", "| stage | acc | letter ECE |", "|---|---|---|"]
    lines += [f"| {k} | {v['macro_acc']:.3f} | {v['macro_ece']:.4f} |" for k, v in summary.items()]
    lines += ["", "## Per subset", "", "| stage | results |", "|---|---|"] + rows
    if final_conf:
        before_conf = (grpo or {}).get("before_conf", {})
        lines += ["", "## Verbalized confidence (GRPO objective)", "",
                  "| subset | acc | mean conf | conf sd | conf AUROC | conf ECE | Brier | conf ECE before GRPO |",
                  "|---|---|---|---|---|---|---|---|"]
        for s, d in sorted(final_conf.items()):
            b = before_conf.get(s, {}).get("conf_ece")
            lines.append(f"| {s} | {d['acc']:.3f} | {d['mean_conf']:.3f} | {d.get('conf_sd', float('nan')):.3f} | "
                         f"{d.get('conf_auroc', float('nan')):.3f} | {d['conf_ece']:.4f} | "
                         f"{d['conf_brier']:.4f} | {'' if b is None else f'{b:.4f}'} |")
    (args.out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
