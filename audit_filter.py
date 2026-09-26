"""Audit the text-reading keyword filter on question text only (no images decoded)."""
import random, re, sys
from datasets import load_dataset
from cauldron_tasks import CAULDRON, CAULDRON_REVISION, READS_TEXT, convert, DESCRIPTIVE

# Broader net to find what the filter misses; not used for filtering.
SUSPECT = re.compile(r"\b(what does|say|name|number|brand|store|team|company|airline|year|price|"
                     r"street|city|letter|what kind of (shop|business|restaurant)|what is on the|"
                     r"what (is|are) (the )?(message|title)|which (way|direction)|how much|cost|"
                     r"channel|station|website|flag|country|advertis|poster|menu|screen|display|"
                     r"monitor|phone|book|magazine|newspaper|shirt|hat)\b", re.I)
N = int(sys.argv[1])
rng = random.Random(0)
import os
for subset in ["vqav2", "cocoqa", "aokvqa", "visual7w", "tallyqa", "vsr", "nlvr2", "iconqa"]:
    ds = load_dataset(CAULDRON, subset, revision=CAULDRON_REVISION, split="train", streaming=True).select_columns(["texts"])
    kept, rejected, total = [], [], 0
    for row in ds:
        for qa in row["texts"]:
            c = convert(subset, qa["user"], qa["assistant"])
            if c is None:
                continue
            total += 1
            q, opts, ans = c
            text = q + " | " + " / ".join(opts)
            (rejected if READS_TEXT.search(q + " " + " ".join(opts)) else kept).append(text)
        if total >= N:
            break
    sus = [k for k in kept if SUSPECT.search(k)]
    m = READS_TEXT
    print(f"\n##### {subset}: {total} usable, rejected {len(rejected)} ({len(rejected)/total:.1%}), kept-but-suspect {len(sus)} ({len(sus)/total:.1%})")
    print("  -- rejected sample (check for false positives):")
    for t in rng.sample(rejected, min(8, len(rejected))):
        print("   R", m.search(t).group(0), "::", t[:140])
    print("  -- kept & suspect sample (check for misses):")
    for t in rng.sample(sus, min(10, len(sus))):
        print("   S", SUSPECT.search(t).group(0), "::", t[:140])
    print("  -- kept random sample:")
    for t in rng.sample(kept, min(10, len(kept))):
        print("   K ::", t[-140:] if subset == "nlvr2" else t[:140])

os._exit(0)
