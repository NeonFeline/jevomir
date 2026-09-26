"""Convert Cauldron subsets into closed-choice items for one-pass letter readout.

Only camera photos or synthetic renders are used, and questions that require reading
text from the image are dropped. Every item becomes:
  {"subset", "image_id", "images", "question", "options": [str], "answer": int}
so all subsets share one prompt format and one readout (option letters).
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from datasets import load_dataset

CAULDRON = "HuggingFaceM4/the_cauldron"
CAULDRON_REVISION = "847a98a779b1652d65111daf20c972dfcd333605"
LETTERS = "ABCDEFGHIJKLMNOP"

# okvqa and clevr_math ship without images; raven draws the answer letters inside
# the image; the rest of the Cauldron is documents, charts, tables, UI or long captions.
SUBSETS = ["clevr", "vqav2", "cocoqa", "aokvqa", "visual7w", "tallyqa", "vsr", "nlvr2", "iconqa"]
PHOTO_SUBSETS = {"vqav2", "cocoqa", "aokvqa", "visual7w", "tallyqa", "vsr", "nlvr2"}

# Phrases that need text, digits or clock faces read off the image. Plain objects
# ("table", "plate", "how many signs", "time of day") are deliberately not matched.
READS_TEXT = re.compile(
    r"\b(written|writing|write|spell|spelled|says?|said|reads?|the text|words?|letters?|"
    r"brand|logo|label|jersey|license plate|website|web ?site|url|sponsor|advertis\w*|"
    r"airline|company|team|what number|which number|number \d+|numbers? (on|of the)|"
    r"on the sign|the sign (say|read|mean)|sign says|street name|name of the|what is the name|"
    r"clocks?|watch shows|what time (is|does|was)|time (is|was) (it|shown|on)|ruler|calendar|"
    r"price|cost|how much|score|scoreboard|channel|title|menu)\b",
    re.I,
)

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
COUNT_OPTIONS = [str(i) for i in range(11)]
CLEVR_VOCAB = [
    ["yes", "no"],
    COUNT_OPTIONS,
    ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    ["cube", "sphere", "cylinder"],
    ["rubber", "metal"],
    ["small", "large"],
]


DESCRIPTIVE = -1


def answer_kind(options):
    if options == ["Yes", "No"]:
        return "yes_no"
    if options == COUNT_OPTIONS:
        return "count"
    return "choice"


def _stem(question: str) -> str:
    return " ".join(question.lower().split()[:2])


def _clean(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^answer:\s*", "", answer, flags=re.I)
    return answer.split("\n")[0].strip().rstrip(".").strip().lower()


def _first_line(user: str) -> str:
    return user.split("\n")[0].strip()


def _yes_no(question, answer):
    if answer in {"yes", "no"}:
        return question, ["Yes", "No"], 0 if answer == "yes" else 1
    return None


def _count(question, answer):
    if answer in NUMBER_WORDS:
        answer = str(NUMBER_WORDS.index(answer))
    if answer in COUNT_OPTIONS:
        return question, COUNT_OPTIONS, int(answer)
    return None


def _lettered_choices(user, answer):
    """visual7w / iconqa: 'Question: ...\\nChoices:\\nA. x\\nB. y\\nAnswer with the letter.'"""
    m = re.match(r"Question:\s*(.*?)\nChoices:\n(.*?)\nAnswer with the letter", user, re.S)
    if not m:
        return None
    options = [re.sub(r"^[A-P]\.\s*", "", line).strip().rstrip(".") for line in m.group(2).split("\n") if line.strip()]
    letter = answer.strip().upper()
    if len(letter) != 1 or letter not in LETTERS[:len(options)]:
        return None
    return m.group(1).strip(), options, LETTERS.index(letter)


def _aokvqa(user, answer):
    m = re.search(r"\nOptions:\s*(.*?)\.?\s*$", user, re.S)
    if not m:
        return None
    options = [o.strip() for o in m.group(1).split(",") if o.strip()]
    matches = [i for i, o in enumerate(options) if o.lower() == answer]
    if len(options) < 2 or len(matches) != 1:
        return None
    return _first_line(user), options, matches[0]


def _claim(user, answer):
    """vsr / nlvr2: keep the claim question, drop the 'Answer yes or no.' instruction."""
    question = re.sub(r"\s*Answer yes or no\.?\s*$", "", user.strip()).strip()
    return _yes_no(question, answer)


def convert(subset: str, user: str, assistant: str):
    """Return (question, options, answer_index) or None if the pair is unusable."""
    answer = _clean(assistant)
    if subset == "clevr":
        question = _first_line(user)
        for vocab in CLEVR_VOCAB:
            if answer in vocab:
                return question, [v.capitalize() if not v.isdigit() else v for v in vocab], vocab.index(answer)
        return None
    if subset in {"vqav2", "cocoqa"}:
        question = _first_line(user)
        # Descriptive answers return a marker; iter_items adds one random distractor.
        return _yes_no(question, answer) or _count(question, answer) or (question, [answer], DESCRIPTIVE)
    if subset == "tallyqa":
        return _count(_first_line(user), answer)
    if subset in {"visual7w", "iconqa"}:
        return _lettered_choices(user, assistant.strip().removeprefix("Answer:").strip())
    if subset == "aokvqa":
        return _aokvqa(user, answer)
    if subset in {"vsr", "nlvr2"}:
        return _claim(user, answer)
    raise ValueError(f"Unknown subset {subset}")


def needs_reading(question: str, options: list[str]) -> bool:
    return bool(READS_TEXT.search(question + " " + " ".join(options)))


def image_key(images) -> str:
    """Perceptual difference hash of the first image, so the same COCO photo shared by
    vqav2/cocoqa/aokvqa/visual7w/vsr lands on the same side of the train/test split."""
    small = images[0].convert("L").resize((9, 8))
    px = list(small.getdata())
    bits = [px[r * 9 + c] > px[r * 9 + c + 1] for r in range(8) for c in range(8)]
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def iter_items(subset: str, limit: int, seed: int = 0, per_image: int = 3):
    """Yield up to `limit` usable items, at most `per_image` per image, in a seeded stream order."""
    ds = load_dataset(CAULDRON, subset, revision=CAULDRON_REVISION, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    rng = random.Random(seed)
    # Answers seen so far, by question stem ("what color", "what is"), for descriptive distractors.
    pools = defaultdict(list)
    produced = 0
    for row_index, row in enumerate(ds):
        images = [img.convert("RGB") for img in row["images"] if img is not None]
        if not images:
            continue
        taken = 0
        key = image_key(images)
        for qa_index, qa in enumerate(row["texts"]):
            converted = convert(subset, qa["user"], qa["assistant"])
            if converted is None:
                continue
            question, options, answer = converted
            kind = answer_kind(options)
            if answer == DESCRIPTIVE:
                correct = options[0]
                pool = pools[_stem(question)]
                others = sorted({a for a in pool if a != correct})
                if correct not in pool:
                    pool.append(correct)
                if not others:
                    continue
                distractor = rng.choice(others)
                options = [correct.capitalize(), distractor.capitalize()]
                if rng.random() < 0.5:
                    options.reverse()
                answer, kind = options.index(correct.capitalize()), "descriptive"
            if not 2 <= len(options) <= len(LETTERS) or needs_reading(question, options):
                continue
            yield {
                "subset": subset,
                "image_id": f"{subset}/{row_index}",
                "image_key": key,
                "qa_index": qa_index,
                "images": images,
                "question": question,
                "options": options,
                "answer": answer,
                "answer_kind": kind,
            }
            produced += 1
            taken += 1
            if produced >= limit:
                return
            if taken >= per_image:
                break
