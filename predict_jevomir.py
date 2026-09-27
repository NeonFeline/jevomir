"""Score a closed-choice question about one or two images with the fine-tuned jevomir model.

Same model, prompt and readout as api_server.py (it calls the server's load() and score()),
without HTTP:

    python predict_jevomir.py scene.png --question "How many cylinders are there?" --options 0 1 2 3
    python predict_jevomir.py left.jpg right.jpg --question "Do both images show dogs?" --options yes no

Prints JSON: per-option probabilities, the prediction, its raw softmax confidence and the
probe's calibrated P(correct). --save-merged DIR also writes the base model with the adapter
merged in (bf16, ~8.5 GB) for serving with api_server.py --model DIR or any HF-compatible stack.
"""

import argparse
import json
import sys
from pathlib import Path

from fastapi import HTTPException
from PIL import Image

import api_server as srv

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=Path, help="one or two images")
    ap.add_argument("--question", default="")
    ap.add_argument("--options", nargs="+", default=[])
    ap.add_argument("--adapter", type=Path, default=ROOT / "artifacts/jevomir-v2/adapter")
    ap.add_argument("--probe", type=Path, default=ROOT / "artifacts/jevomir-v2/head")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B", help="base model the adapter was trained on")
    ap.add_argument("--save-merged", type=Path, default=None, help="write the merged model here")
    args = ap.parse_args()
    if not args.save_merged and not (args.images and args.question and args.options):
        ap.error("give images, --question and --options (or --save-merged DIR)")
    if not 0 <= len(args.images) <= srv.MAX_IMAGES:
        ap.error(f"at most {srv.MAX_IMAGES} images")

    srv.load(args.model, args.adapter, args.probe)
    if args.save_merged:
        args.save_merged.mkdir(parents=True, exist_ok=True)
        srv.State.model.save_pretrained(args.save_merged)
        srv.State.processor.save_pretrained(args.save_merged)
        print(f"merged model -> {args.save_merged}", file=sys.stderr)
    if args.images:
        images = [Image.open(p).convert("RGB") for p in args.images]
        try:
            result = srv.score(args.question, args.options, images)
        except HTTPException as error:
            raise SystemExit(f"error: {error.detail}") from None
        for key in ("model", "probe", "notes"):
            result.pop(key)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
