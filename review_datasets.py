"""Audit and clean closed-choice calibration items.

Item schema: {"subset", "item_id", "question", "options": [str], "answer": int,
              "answer_kind": str, "images": [PIL, ...] (optional, may be empty)}

Audit reports duplicate questions, option-sanity violations, answer-position
imbalance, question/option lengths, and answer leakage into the question.
clean() applies hard filters, dedupes, and shuffles option order deterministically
so the answer index carries no positional signal.

CLI: python review_datasets.py path/to/items.jsonl [--out cleaned.jsonl] [--report report.json]
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def _dup_key(it):
    return _norm(it["question"])


def _leak(it):
    """Answer option text (leading name before ':', and basename for paths) in the question."""
    ans = it["options"][it["answer"]]
    q = _norm(it["question"])
    needles = [_norm(ans.split(":")[0]) if ":" in ans else _norm(ans)]
    if "/" in ans:
        base = _norm(ans.strip().rsplit("/", 1)[-1])
        if base not in needles:
            needles.append(base)
    for needle in needles:
        if len(needle) >= 4 and re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", q):
            return True
    return False


def _pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))] if values else 0


def audit(items):
    n = len(items)
    dup_q = Counter(_dup_key(it) for it in items)
    dup_qo = Counter((_dup_key(it), tuple(_norm(o) for o in it["options"])) for it in items)
    qlen = [len(it["question"]) for it in items]
    olen = [len(o) for it in items for o in it["options"]]
    return {
        "n": n,
        "dup_questions": n - len(dup_q),
        "dup_question_option_sets": n - len(dup_qo),
        "empty_options": sum(any(not o.strip() for o in it["options"]) for it in items),
        "duplicate_options_in_item": sum(
            len({_norm(o) for o in it["options"]}) != len(it["options"]) for it in items),
        "bad_answer_index": sum(not (0 <= it["answer"] < len(it["options"])) for it in items),
        "answer_leak_into_question": sum(_leak(it) for it in items),
        "answer_index_hist": dict(sorted(Counter(it["answer"] for it in items).items())),
        "option_count_hist": dict(sorted(Counter(len(it["options"]) for it in items).items())),
        "q_len_p50": _pct(qlen, 0.5), "q_len_p90": _pct(qlen, 0.9), "q_len_max": max(qlen, default=0),
        "opt_len_p50": _pct(olen, 0.5), "opt_len_p90": _pct(olen, 0.9), "opt_len_max": max(olen, default=0),
    }


def clean(items, subset, max_question=4000, max_option=300, min_options=2, max_options=16,
          drop_leak=True, dedup=True, dedup_by_question=False, shuffle_options=True, seed=0):
    """Hard filters + dedup + deterministic option shuffle (answer follows)."""
    kept, seen, seen_q, reasons = [], set(), set(), Counter()
    for it in items:
        if not (min_options <= len(it["options"]) <= max_options):
            reasons["option_count"] += 1
            continue
        if any(not o.strip() or len(o) > max_option for o in it["options"]):
            reasons["bad_option_text"] += 1
            continue
        if len(it["question"]) > max_question:
            reasons["question_too_long"] += 1
            continue
        if not (0 <= it["answer"] < len(it["options"])):
            reasons["bad_answer_index"] += 1
            continue
        if len({_norm(o) for o in it["options"]}) != len(it["options"]):
            reasons["duplicate_options"] += 1
            continue
        if drop_leak and _leak(it):
            reasons["answer_leak"] += 1
            continue
        key = (_dup_key(it), tuple(_norm(o) for o in it["options"]))
        if dedup and key in seen:
            reasons["duplicate_item"] += 1
            continue
        if dedup_by_question and _dup_key(it) in seen_q:
            reasons["duplicate_question"] += 1
            continue
        seen.add(key)
        seen_q.add(_dup_key(it))
        options, answer = list(it["options"]), it["answer"]
        if shuffle_options:
            rng = random.Random(f"{seed}:{it['item_id']}")
            order = list(range(len(options)))
            rng.shuffle(order)
            options = [options[j] for j in order]
            answer = order.index(answer)
        kept.append({**it, "subset": subset, "options": options, "answer": answer})
    report = audit(kept)
    report["kept"] = len(kept)
    report["dropped"] = len(items) - len(kept)
    report["dropped_reasons"] = dict(sorted(reasons.items()))
    return kept, report


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(items, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            it = {k: v for k, v in it.items() if k != "images"}
            f.write(json.dumps(it) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--max-question", type=int, default=4000)
    ap.add_argument("--no-leak-filter", action="store_true")
    args = ap.parse_args()

    all_report = {}
    for path in args.inputs:
        items = load_jsonl(path)
        cleaned, report = clean(items, path.stem, max_question=args.max_question,
                                drop_leak=not args.no_leak_filter)
        all_report[path.stem] = report
        print(f"== {path.stem}: raw {report['n']} -> kept {report['kept']} (dropped {report['dropped']})")
        print(json.dumps({k: report[k] for k in ("dup_questions", "dup_question_option_sets",
                                                 "empty_options", "duplicate_options_in_item",
                                                 "answer_leak_into_question", "bad_answer_index",
                                                 "answer_index_hist", "option_count_hist",
                                                 "q_len_p50", "q_len_p90", "q_len_max",
                                                 "opt_len_p50", "opt_len_p90", "opt_len_max")}, indent=2))
        if args.out:
            out = args.out / f"{path.stem}.clean.jsonl" if args.out.is_dir() or args.out.suffix == "" else args.out
            write_jsonl(cleaned, out)
            print("wrote", out)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(all_report, indent=2))


if __name__ == "__main__":
    main()
