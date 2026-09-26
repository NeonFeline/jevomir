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
