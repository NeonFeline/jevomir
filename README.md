# jevomir

VLM probability calibration experiments: does the model's stated probability match its
empirical accuracy? Task: "Does this image contain a cat?" on Oxford-IIIT Pets, with
Qwen2.5-VL-3B-Instruct and Qwen3.5-4B.

Two pipelines:
- **generation** (`extract.py`): model generates a free-form answer; "jev" probability =
  `exp(mean log p_i)` over the answer tokens; multi-layer hidden states are pooled from the
  answer tokens.
- **one forward pass** (`extract_onepass.py`): no tokens are generated. Forced-choice
  protocols (cat/dog, yes/no) are scored from the logits at the final prompt position;
  prompt-end hidden states are used.

`train_probe.py` trains a small calibration probe (per-layer linear projection + MLP) to
predict P(answer correct) from hidden states of layers selected automatically, and reports
ECE / Brier / AUROC plus reliability diagrams on a held-out image split.

## Files
- `extract.py`, `extract_onepass.py` - feature extraction (answers, probabilities, hidden states)
- `train_probe.py` - probe training + calibration metrics + reliability plots
- `predict.py`, `predict_onepass.py` - end-to-end single-image demos
- `artifacts/` - metrics.json, summary.json, reliability.png, samples/, trained `probe.pt`
  (multi-GB `features_*.pt` tensors are git-ignored; regenerate with the extract scripts)
- `requirements.txt` - frozen uv environment

## Run
```bash
uv venv .venv && uv pip install -r requirements.txt
python extract_onepass.py --split train --limit 1500 --batch 8 --out artifacts/q35/features_train.pt
python extract_onepass.py --split test  --limit 1500 --batch 8 --out artifacts/q35/features_test.pt
python train_probe.py --train artifacts/q35/features_train.pt --test artifacts/q35/features_test.pt --out artifacts/q35 --d-proj 128
python predict_onepass.py path/to/image.jpg
```

## Cauldron probe (Qwen3.5-4B, many subsets)

`cauldron_tasks.py` turns [the Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
(revision `847a98a`) into closed-choice items, using only camera photos or synthetic renders:
`clevr`, `vqav2`, `cocoqa`, `aokvqa`, `visual7w`, `tallyqa`, `vsr`, `nlvr2`, `iconqa`.
Excluded: documents/charts/tables/UI/OCR subsets, long-caption subsets, `okvqa` and
`clevr_math` (no images in the Cauldron), and `raven` (answer letters are drawn in the image).
Questions that need text, digits or clock faces read off the image are dropped by a keyword
filter (`READS_TEXT`); `audit_filter.py` prints rejected and kept samples per subset.
Descriptive vqav2/cocoqa answers get one random distractor drawn from answers to questions
with the same two-word stem; yes/no and counts use fixed option sets.

Every item is scored in one forward pass by option-letter logits. Images are letterboxed to
448x448 and sequences padded to a multiple of 128, because the linear-attention Triton kernels
re-tune on every new shape (about 5x throughput on an H100). The probe is split by a perceptual
image hash, so shared COCO photos never straddle train/test, and `vsr` is held out entirely.

```bash
pip install flash-linear-attention   # fast Qwen3.5 kernels; needs python3.x-dev headers
python audit_filter.py 2000
python download_cauldron.py
CUDA_VISIBLE_DEVICES=0 python extract_cauldron.py --out runs/cauldron-001 --limit 50000 --batch 64 --workers 12
python train_probe_cauldron.py --features runs/cauldron-001 --out runs/probe-001 --holdout-subsets vsr
```

### Scoring API

`api_server.py` serves the model and the Cauldron probe over HTTP with an API key; see
[API.md](API.md). Start it with `JEVOMIR_API_KEY=<secret> python api_server.py --probe runs/probe-002`
and expose the local port with a tunnel such as `ngrok http 8100`.

`web/` is a browser UI for the API (image upload, question, options, results); run
`python web/server.py` and see [web/README.md](web/README.md) for how it integrates.
