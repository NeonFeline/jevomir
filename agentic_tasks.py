"""Convert agentic datasets into closed-choice calibration items (text-only).

- BFCL v3 single-call items -> "which function should be called?": options are the
  candidate functions of the same item (name + description), answer = ground-truth call.
- SWE-bench Verified -> "which file needs the fix?": answer = most-changed gold file,
  distractors = test files of the same instance and same-repo files, never a gold file.

Every item goes through review_datasets.clean (dedup, leakage, lengths, option shuffle)
so the emitted jsonl is the cleaned set. Text-only items carry no images; the extraction
path scores them in one forward pass like the Cauldron items.

CLI:
  python agentic_tasks.py --swe-parquet ~/agentic/swebench/data/test-00000-of-00001.parquet \
      --bfcl-parquet ~/agentic/bfcl_v3/data/train-00000-of-00001.parquet \
      --out runs/agentic --limit 4000
"""

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from review_datasets import _norm, clean, write_jsonl

DIFF_RE = re.compile(r"^diff --git a/(.+?) b/", re.M)
SIMPLE_ID = re.compile(r"^(live_)?simple_\d")


def _changed_lines(patch: str):
    """{file: changed-line count} from a unified diff (no repo checkout needed)."""
    files, cur = defaultdict(int), None
    for line in patch.splitlines():
        m = DIFF_RE.match(line)
        if m:
            cur = m.group(1)
            files.setdefault(cur, 0)
            continue
        if cur and line[:1] in "+-" and not line.startswith(("+++", "---")):
            files[cur] += 1
    return files


def _function_pool(rows):
    """Global pool of function specs (name -> spec) from every row that parses."""
    pool = {}
    for r in rows:
        try:
            funcs = json.loads(r["function"])
        except (TypeError, ValueError):
            continue
        for f in funcs:
            if f.get("name") and f["name"] not in pool:
                pool[f["name"]] = f
    return pool


def build_bfcl(rows, limit=None, n_distractors=5, seed=0):
    """Single-call items -> pick the right function. Distractors come from the global pool."""
    pool = _function_pool(rows)
    items, seen_q = [], set()
    for r in rows:
        iid = r["id"]
        if not SIMPLE_ID.match(iid):
            continue
        try:
            gt = json.loads(r["ground_truth"])
            msgs = json.loads(r["chat_completion_input"])
        except (TypeError, ValueError):
            continue
        if len(gt) != 1 or not isinstance(gt[0], dict):
            continue
        correct = next(iter(gt[0]))
        if correct not in pool:
            continue
        user = [m["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)]
        if not user:
            continue
        question = user[-1].strip()
        if _norm(question) in seen_q:
            continue
        seen_q.add(_norm(question))
        others = sorted(name for name in pool if name != correct)
        rng = random.Random(f"{seed}:{iid}")
        distractors = rng.sample(others, min(n_distractors, len(others)))
        ordered = [pool[correct]] + [pool[name] for name in distractors]
        if len(ordered) < 2:
            continue
        options = [f"{f['name']}: {re.sub(r'[ \t\r\n]+', ' ', f.get('description') or '').strip()[:200]}"
                   for f in ordered]
        items.append({
            "subset": "bfcl_selection", "item_id": iid, "question": question,
            "options": options, "answer": 0, "answer_kind": "choice", "images": [],
        })
        if limit and len(items) >= limit:
            break
    return items


def build_swebench(rows, limit=None, max_changed=600, max_gold_files=5):
    repo_files = defaultdict(set)
    for r in rows:
        repo_files[r["repo"]].update(_changed_lines(r["test_patch"] or ""))

    items = []
    for r in rows:
        gold = _changed_lines(r["patch"] or "")
        if not gold or sum(gold.values()) > max_changed or len(gold) > max_gold_files:
            continue
        answer_file = max(gold, key=gold.get)
        answer_dir = answer_file.rsplit("/", 1)[0] if "/" in answer_file else ""
        pool = sorted(f for f in repo_files[r["repo"]] if f not in gold)
        same_dir = [f for f in pool if f.rsplit("/", 1)[0] == answer_dir]
        other = [f for f in pool if f not in same_dir]
        picks = same_dir[:4] + other[:3]
        if not picks:
            continue
        items.append({
            "subset": "swebench_file", "item_id": r["instance_id"],
            "question": r["problem_statement"].strip(),
            "options": [answer_file] + picks, "answer": 0, "answer_kind": "choice", "images": [],
        })
        if limit and len(items) >= limit:
            break
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swe-parquet", type=Path, default=Path.home() / "agentic/swebench/data/test-00000-of-00001.parquet")
    ap.add_argument("--bfcl-parquet", type=Path, default=Path.home() / "agentic/bfcl_v3/data/train-00000-of-00001.parquet")
    ap.add_argument("--out", type=Path, default=Path("runs/agentic"))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    builders = []
    if args.bfcl_parquet.exists():
        builders.append(("bfcl_selection", build_bfcl, args.bfcl_parquet))
    else:
        print("missing", args.bfcl_parquet)
    if args.swe_parquet.exists():
        builders.append(("swebench_file", build_swebench, args.swe_parquet))
    else:
        print("missing", args.swe_parquet)

    for subset, builder, parquet in builders:
        rows = pq.read_table(parquet).to_pylist()
        raw = builder(rows, limit=args.limit or None)
        cleaned, report = clean(raw, subset, seed=args.seed, dedup_by_question=True)
        write_jsonl(cleaned, args.out / f"{subset}.jsonl")
        write_jsonl(raw, args.out / f"{subset}.raw.jsonl")
        print(f"== {subset}: raw {len(raw)} -> clean {len(cleaned)}")
        print(json.dumps({k: report[k] for k in ("dup_questions", "duplicate_options_in_item",
                                                 "answer_leak_into_question", "bad_answer_index",
                                                 "dropped_reasons", "answer_index_hist",
                                                 "option_count_hist", "q_len_p50", "q_len_p90",
                                                 "q_len_max", "opt_len_p50", "opt_len_p90",
                                                 "opt_len_max")}, indent=2))


if __name__ == "__main__":
    main()
