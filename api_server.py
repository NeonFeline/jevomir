"""HTTP API: image(s) + question + options -> option probabilities + calibrated P(correct).

One forward pass per request with the same prompt, image letterboxing and readout as
extract_cauldron.py, then the calibration probe trained on its hidden states.

    JEVOMIR_API_KEY=... python api_server.py --probe runs/probe-002 --port 8100
    JEVOMIR_API_KEY=... python api_server.py --model runs/X/final-rft1/model --probe runs/X/head
    JEVOMIR_API_KEY=... python api_server.py --adapter artifacts/jevomir-v2/adapter --probe artifacts/jevomir-v2/head

Auth: `Authorization: Bearer <key>` or `X-API-Key: <key>` on every /v1 endpoint.
Interactive schema: /docs (no key needed to read it; every call still needs the key).
"""

import argparse
import base64
import binascii
import hmac
import io
import json
import os
import tempfile
import threading
import time
from pathlib import Path

import torch
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from cauldron_tasks import LETTERS
from extract_cauldron import MODEL_REVISION, forward_batch, letter_ids, load_model, prepare
from train_probe import CalibProbe

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGES = 2
MAX_QUESTION_CHARS = 2000
MAX_OPTION_CHARS = 300
Image.MAX_IMAGE_PIXELS = 50_000_000  # reject decompression bombs

NOTES = [
    "probabilities: softmax over the option letters only, from one forward pass; uncalibrated",
    "calibrated_p_correct: probe estimate that the predicted option is correct; calibrated on the "
    "Cauldron photo/synthetic mix (held-out test ECE 0.010), less reliable on other image types, "
    "text-reading questions or very different question styles (unseen-subset ECE 0.048)",
]


def probe_notes(probe_dir):
    """NOTES with the held-out ECE of the probe actually being served (from its metrics.json)."""
    try:
        m = json.loads((probe_dir / "metrics.json").read_text())
        test_in = m["test_in"]["probe"]["ece"]
        hold = m.get("test_holdout", {}).get("probe", {}).get("ece")
    except (OSError, KeyError, TypeError, ValueError):
        return NOTES
    unseen = f" (unseen-subset ECE {hold:.3f})" if hold is not None else ""
    return [NOTES[0],
            "calibrated_p_correct: probe estimate that the predicted option is correct; calibrated on the "
            f"Cauldron photo/synthetic mix (held-out test ECE {test_in:.3f}), less reliable on other image "
            f"types, text-reading questions or very different question styles{unseen}"]


def load_adapter_model(adapter_dir, base_id, revision):
    """Base model + LoRA adapter merged in bf16, exactly as extract_finetuned.py merges it (so a
    probe trained on those features sees the same hidden states). An adapter too big for one git
    file is stored as adapter_model-0000i-of-0000n.safetensors shards and joined here."""
    from peft import PeftModel
    from safetensors.torch import load_file, save_file
    model, processor = load_model(base_id, revision=revision)
    shards = sorted(adapter_dir.glob("adapter_model-*-of-*.safetensors"))
    if (adapter_dir / "adapter_model.safetensors").exists() or not shards:
        return PeftModel.from_pretrained(model, adapter_dir).merge_and_unload().eval(), processor
    with tempfile.TemporaryDirectory() as tmp:
        tensors = {}
        for shard in shards:
            tensors.update(load_file(shard))
        save_file(tensors, Path(tmp) / "adapter_model.safetensors")
        (Path(tmp) / "adapter_config.json").write_text((adapter_dir / "adapter_config.json").read_text())
        model = PeftModel.from_pretrained(model, tmp).merge_and_unload().eval()
    return model, processor


class ScoreRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    options: list[str] = Field(min_length=2, max_length=len(LETTERS))
    images: list[str] = Field(min_length=1, max_length=MAX_IMAGES,
                              description="base64 image bytes (JPEG/PNG/WebP), optionally as a data: URL")
    id: str | None = Field(default=None, max_length=200)


class State:
    model = processor = probe = ckpt = layers = ids = None
    image_size = 448
    pad_multiple = 128
    lock = threading.Lock()
    info = {}
    notes = NOTES


app = FastAPI(title="jevomir scoring API", version="1.0",
              description="Closed-choice visual questions scored in one forward pass of Qwen3.5-4B, "
                          "with a calibrated probability that the chosen option is correct.")


def require_key(authorization: str | None = Header(default=None), x_api_key: str | None = Header(default=None)):
    expected = os.environ["JEVOMIR_API_KEY"]
    given = x_api_key or (authorization[7:] if authorization and authorization.startswith("Bearer ") else None)
    if not given or not hmac.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


def decode_image(data: bytes) -> Image.Image:
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"Image larger than {MAX_IMAGE_BYTES} bytes")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except Exception as error:  # PIL raises many types for bad input
        raise HTTPException(status_code=422, detail=f"Could not decode image: {error}") from None


def decode_base64(text: str) -> bytes:
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=422, detail="images must be valid base64") from None


def score(question, options, images, request_id=None):
    options = [o.strip() for o in options]
    if any(not o or len(o) > MAX_OPTION_CHARS for o in options):
        raise HTTPException(status_code=422, detail=f"Options must be 1-{MAX_OPTION_CHARS} characters")
    if len(set(options)) != len(options):
        raise HTTPException(status_code=422, detail="Options must be unique")
    started = time.perf_counter()
    item = {"question": question.strip(), "options": options, "images": images}
    with State.lock:  # one GPU; the processor/tokenizer is also not shared across threads
        inputs = prepare(State.processor, [item], State.image_size, State.pad_multiple)
        prepared = time.perf_counter()
        logits, states = forward_batch(State.model, dict(inputs), State.layers, State.ids)
        torch.cuda.synchronize()
    forwarded = time.perf_counter()
    probs = torch.softmax(logits[0, :len(options)].float(), dim=-1)
    pred = int(probs.argmax())
    x = (states.float() - State.ckpt["mu"]) / State.ckpt["sd"]
    with torch.no_grad():
        # Capped: a saturated float32 sigmoid would report certainty the probe cannot have.
        p_correct = min(max(float(torch.sigmoid(State.probe(x))[0]), 0.001), 0.999)
    return {
        "id": request_id,
        "options": [{"letter": LETTERS[i], "text": o, "probability": float(p)}
                    for i, (o, p) in enumerate(zip(options, probs.tolist()))],
        "prediction": {"index": pred, "letter": LETTERS[pred], "text": options[pred]},
        "raw_confidence": float(probs[pred]),
        "calibrated_p_correct": p_correct,
        "input_tokens": int(inputs["attention_mask"].sum()),
        "timing": {"preprocess_s": prepared - started, "forward_s": forwarded - prepared,
                   "total_s": time.perf_counter() - started},
        "model": State.info["model"],
        "probe": State.info["probe"],
        "notes": State.notes,
    }


@app.get("/health")
def health():
    return {"status": "ok" if State.model is not None else "loading"}


@app.get("/v1/info", dependencies=[Depends(require_key)])
def info():
    return State.info | {"limits": {"images": MAX_IMAGES, "image_bytes": MAX_IMAGE_BYTES,
                                    "options": [2, len(LETTERS)], "question_chars": MAX_QUESTION_CHARS,
                                    "option_chars": MAX_OPTION_CHARS}, "notes": State.notes}


@app.post("/v1/score", dependencies=[Depends(require_key)])
def score_json(request: ScoreRequest):
    images = [decode_image(decode_base64(b)) for b in request.images]
    return score(request.question, request.options, images, request.id)


@app.post("/v1/score/upload", dependencies=[Depends(require_key)])
def score_upload(question: str = Form(..., max_length=MAX_QUESTION_CHARS),
                 options: list[str] = Form(..., description="repeat the field once per option"),
                 images: list[UploadFile] = File(...), id: str | None = Form(default=None, max_length=200)):
    if not 2 <= len(options) <= len(LETTERS):
        raise HTTPException(status_code=422, detail=f"Send 2-{len(LETTERS)} options")
    if not 1 <= len(images) <= MAX_IMAGES:
        raise HTTPException(status_code=422, detail=f"Send 1-{MAX_IMAGES} images")
    decoded = [decode_image(f.file.read(MAX_IMAGE_BYTES + 1)) for f in images]
    return score(question, options, decoded, id)


def load(model_id, adapter, probe_dir):
    """Fill State: base model (or merged dir) + optional LoRA adapter, and the calibration probe.
    predict_jevomir.py calls this too, so the CLI and the API score identically."""
    local = Path(model_id).is_dir()
    revision = None if local else MODEL_REVISION
    if adapter:
        State.model, State.processor = load_adapter_model(adapter, model_id, revision)
    else:
        State.model, State.processor = load_model(model_id, revision=revision)
    State.notes = probe_notes(probe_dir)
    State.ids = letter_ids(State.processor.tokenizer)
    State.ckpt = torch.load(probe_dir / "probe.pt", map_location="cpu", weights_only=False)
    State.layers = State.ckpt["layers"]
    State.probe = CalibProbe(State.ckpt["n_layers"], State.ckpt["d_model"], d_proj=State.ckpt["d_proj"])
    State.probe.load_state_dict({k: v.cpu() for k, v in State.ckpt["state_dict"].items()})
    State.probe.eval()
    State.info = {
        "model": {"source": model_id, "revision": revision, "dtype": "bfloat16",
                  "adapter": f"{adapter.parent.name}/{adapter.name}" if adapter else None,
                  "image_size": State.image_size, "layers": State.layers},
        "probe": {"run": probe_dir.name if probe_dir.name != "head" else f"{probe_dir.parent.name}/head"},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True, help="run directory holding probe.pt and metrics.json")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B",
                    help="HF id or a merged fine-tuned model directory; the probe must be trained on this model")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="LoRA adapter directory merged into --model at start (e.g. artifacts/jevomir-v2/adapter)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    args = ap.parse_args()
    if len(os.environ.get("JEVOMIR_API_KEY", "")) < 32:
        raise SystemExit("Set JEVOMIR_API_KEY to a random secret of at least 32 characters")

    load(args.model, args.adapter, args.probe)
    # Compile/tune kernels for the common one-image shape before accepting traffic.
    score("Is this a test?", ["Yes", "No"], [Image.new("RGB", (64, 64), (128, 128, 128))])
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
