"""Robo2VLM-1 (OXE robot-manipulation MCQ) -> cleaned closed-choice items + review.

Robo2VLM-1 rows are {id, question, choices, correct_answer, image}; the schema already
matches our closed-choice format, so the work here is cleanliness:

- text filters via review_datasets.clean (option sanity, lengths, leakage, dedup, shuffle)
- image-aware dedup: same image + same question never appears twice, and image keys are
  tracked so the probe trainer's hash split cannot leak the same scene across train/test
- cross-split overlap report between the provided parquet files

CLI:
  python robo2vlm_tasks.py --parquet ~/agentic/robo2vlm/data/train-00000-of-00262.parquet \
      --parquet ~/agentic/robo2vlm/data/test-00000-of-00003.parquet \
      --out runs/agentic --limit 0
"""

import argparse
import ast
import io
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from cauldron_tasks import image_key
from review_datasets import _norm, clean, write_jsonl


def load_rows(path):
    return pq.read_table(path).to_pylist()


def to_items(rows, subset="robo2vlm"):
    items, bad = [], Counter()
    for r in rows:
        img = r.get("image") or {}
        data = img.get("bytes")
        if not data:
            bad["missing_image"] += 1
            continue
        choices = r.get("choices")
        if isinstance(choices, str):
            try:
                choices = ast.literal_eval(choices)
            except (ValueError, SyntaxError):
                bad["unparsable_choices"] += 1
                continue
        if not isinstance(choices, (list, tuple)) or not choices:
            bad["empty_choices"] += 1
            continue
        try:
            ans = int(r.get("correct_answer"))
        except (TypeError, ValueError):
            bad["bad_answer"] += 1
            continue
        if not (0 <= ans < len(choices)):
            bad["bad_answer"] += 1
            continue
        items.append({
            "subset": subset, "item_id": r["id"], "question": str(r["question"]).strip(),
            "options": [str(c).strip() for c in choices], "answer": ans,
            "answer_kind": "choice", "image_bytes": data,
        })
    return items, bad


def image_stats(items):
    sizes, n = [], 0
    for it in items:
        try:
            im = Image.open(io.BytesIO(it["image_bytes"]))
            sizes.append(im.size)
            n += 1
        except Exception:
            continue
    if not sizes:
        return {}
    ws = sorted(w for w, _ in sizes)
    hs = sorted(h for _, h in sizes)
    return {"n_images": n, "w_p50": ws[len(ws) // 2], "h_p50": hs[len(hs) // 2],
            "w_max": ws[-1], "h_max": hs[-1]}


def dedup_by_image_question(items, seen_global=None):
    """Drop repeated (image, question, options) triples, then attach perceptual image keys.

    `seen_global` carries signatures across parquet files so the same image/question
    cannot appear in both the train and test shards.
    """
    seen = seen_global if seen_global is not None else set()
    kept, keys = [], []
    for it in items:
        try:
            im = Image.open(io.BytesIO(it["image_bytes"])).convert("RGB")
        except Exception:
            continue
        key = image_key([im])
        sig = (key, _norm(it["question"]), tuple(_norm(o) for o in it["options"]))
        if sig in seen:
            continue
        seen.add(sig)
        keys.append(key)
        kept.append({**it, "image_key": key})
    return kept, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/agentic"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-question", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_keys, summaries = {}, {}
    seen_global = set()
    for path in args.parquet:
        rows = load_rows(path)
        raw, bad = to_items(rows)
        if args.limit:
            raw = raw[:args.limit]
        deduped, keys = dedup_by_image_question(raw, seen_global)
        cleaned, report = clean(deduped, subset=deduped[0]["subset"] if deduped else "robo2vlm",
                                max_question=args.max_question, seed=args.seed,
                                dedup=False, dedup_by_question=False)
        index = []
        for it in cleaned:
            row = {k: v for k, v in it.items() if k != "image_bytes"}
            row["parquet"] = path.name
            index.append(row)
        write_jsonl(index, args.out / f"robo2vlm_{path.stem}.index.jsonl")
        all_keys[path.stem] = {it["image_key"] for it in cleaned}
        summaries[path.stem] = {
            "rows": len(rows), "items": len(raw), "bad_rows": dict(bad),
            "duplicate_image_question": len(raw) - len(deduped),
            "text_report": {k: report[k] for k in ("dup_questions", "duplicate_options_in_item",
                                                   "answer_leak_into_question", "bad_answer_index",
                                                   "dropped_reasons", "answer_index_hist",
                                                   "option_count_hist", "q_len_p50", "q_len_p90",
                                                   "q_len_max", "opt_len_p50", "opt_len_p90",
                                                   "opt_len_max")},
            **image_stats(deduped),
        }
        print(f"== {path.stem}: rows {len(rows)} -> items {len(raw)} -> clean {len(cleaned)} "
              f"(dup image+question {len(raw) - len(deduped)}, dropped {report['dropped']})")
        print(json.dumps({**{k: summaries[path.stem][k] for k in ("bad_rows", "duplicate_image_question")},
                          **summaries[path.stem]["text_report"],
                          **{k: summaries[path.stem].get(k) for k in ("n_images", "w_p50", "h_p50")}}, indent=2))
        if not index:
            print("  cleaned item count is 0, re-run review on the raw items")

    names = sorted(all_keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = len(all_keys[names[i]] & all_keys[names[j]])
            if inter:
                print(f"image overlap {names[i]} <-> {names[j]}: {inter}")
    (args.out / "robo2vlm_review.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
