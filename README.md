# jevomir

VLM probability calibration experiments: does the model's stated probability match its
empirical accuracy? Task: "Does this image contain a cat?" on Oxford-IIIT Pets, with
Qwen2.5-VL-3B-Instruct and Qwen3.5-4B.

The current fine-tuned model is **jevomir-v2**; see [below](#jevomir-v2-fine-tuned-model) for
what it is, how well it does, and how to run and serve it.

## jevomir-v2 (fine-tuned model)

Qwen3.5-4B (revision `851bf6e`) with a LoRA adapter (r=16, alpha=32), fine-tuned to answer
closed-choice visual questions with **calibrated** option probabilities, plus a calibration
head that turns hidden states into P(the chosen answer is correct). Each question is scored in
one forward pass: the model reads the image(s), the question and lettered options, and the
softmax over the option-letter logits is the answer distribution. Nothing is generated.

Training ran three stages on the Cauldron closed-choice mix (`clevr`, `vqav2`, `tallyqa`,
`nlvr2`, `iconqa`). Each stage used its own 16k items, and no image was shared across stages:
1. **SFT** on the answer letter (`train_qlora.py`).
2. **RFT**, two rounds of rejection sampling + SFT (`train_rft.py`).
3. **Policy stage** (`train_grpo.py`), which trains the full answer distribution with a proper
   score: `-log p(answer) + 0.5 KL(p_permuted || p) + 0.1 KL(p || p_RFT)`. The first term
   rewards calibration, the second rewards the same answer when the options are shuffled, and
   the third keeps the model close to RFT. The file's docstring explains why this replaced the
   earlier sampled-GRPO objective.

### Results (held out)

7,500 test items, 1,500 per subset. The split is by perceptual image hash, so no test image
appears in any training stage. ECE is the expected calibration error of the top option's
probability; lower is better. The full report, with paired bootstrap CIs, McNemar tests and
examples of changed answers, is in [artifacts/jevomir-v2/EVAL.md](artifacts/jevomir-v2/EVAL.md)
(`eval_compare.py`).

| subset | base acc | RFT acc | **v2 acc** | base ECE | RFT ECE | **v2 ECE** |
|---|---|---|---|---|---|---|
| clevr | 0.895 | 0.988 | **0.987** | 0.026 | 0.006 | **0.005** |
| iconqa | 0.857 | 0.976 | **0.987** | 0.029 | 0.011 | **0.005** |
| nlvr2 | 0.847 | 0.917 | **0.923** | 0.036 | 0.057 | **0.032** |
| tallyqa | 0.717 | 0.759 | **0.763** | 0.046 | 0.104 | **0.049** |
| vqav2 | 0.897 | 0.924 | **0.929** | 0.050 | 0.033 | **0.013** |
| **all** | 0.843 | 0.913 | **0.918** | 0.028 | 0.038 | **0.016** |

Against RFT, v2 is +0.5 pt accuracy (95% CI +0.2 to +0.8; 105 answers fixed vs 66 broken,
McNemar p = 0.004) and has 60% lower ECE. NLL and Brier improve on every subset. It also
picks the same answer when the options are shuffled 98.3% of the time. RFT alone had made the
model overconfident (tallyqa ECE 0.046 → 0.104); the policy stage brings calibration back to
the base model's level while keeping the accuracy gain.

The calibration head (`artifacts/jevomir-v2/head`) is trained on v2's hidden states. On
held-out items its P(correct) has ECE 0.014 and AUROC 0.92 ([reliability
plot](artifacts/jevomir-v2/head/reliability_test_in.png)). Like the model, it is calibrated for
Cauldron-style photos and synthetic renders. Expect it to be less reliable on documents,
charts, text-reading questions or very different question styles.

### Files

```
artifacts/jevomir-v2/
  adapter/   adapter_config.json + adapter_model-0000{1,2}-of-00002.safetensors
             (the fp32 LoRA weights, split in two to stay under GitHub's 100 MB file limit;
              api_server.py / predict_jevomir.py join them automatically)
  head/      probe.pt (calibration head), metrics.json, reliability_test_in.png
  EVAL.md    held-out comparison base vs RFT vs v2; eval.json has the numbers
```

The base model (about 8.5 GB) is not in the repo. It is downloaded from Hugging Face
(`Qwen/Qwen3.5-4B` @ `851bf6e`) on first use.

### Install

A CUDA GPU that fits the 4B model in bf16 (about 9 GB of weights plus activations) is enough;
the model was developed on A100/H100 80 GB.

```bash
uv venv .venv && uv pip install -r requirements.txt   # pulls CUDA 12.8 torch wheels
uv pip install fastapi uvicorn python-multipart        # only for the HTTP API
uv pip install flash-linear-attention                  # optional: fast Qwen3.5 kernels (needs python3.x-dev)
```

### Run it on an image (CLI)

```bash
python predict_jevomir.py scene.png --question "How many cylinders are there?" --options 0 1 2 3 4 5
python predict_jevomir.py left.jpg right.jpg \
    --question "The left image shows exactly two dogs. Is that true?" --options Yes No
```

This prints JSON with each option's probability, the `prediction`, its `raw_confidence` (the
top softmax probability) and `calibrated_p_correct` (the head's estimate). It uses the same
code path as the API. Options become letters A, B, C, … in the order given, and 2 to 16
options are allowed. Use one image, or two for questions about a pair.

### Serve it over HTTP

```bash
export JEVOMIR_API_KEY=$(python -c "import secrets; print('jev_' + secrets.token_urlsafe(32))")
python api_server.py --adapter artifacts/jevomir-v2/adapter --probe artifacts/jevomir-v2/head \
    --host 0.0.0.0 --port 8100
```

At start the server downloads the base model if needed, merges the adapter, loads the head and
warms up the kernels. `GET /health` returns `{"status": "ok"}` once it is ready. Then:

```bash
printf '{"question":"How many cylinders are there?","options":["0","1","2","3"],"images":["%s"]}' \
  "$(base64 -w0 scene.png)" > req.json
curl -X POST http://localhost:8100/v1/score -H "Authorization: Bearer $JEVOMIR_API_KEY" \
  -H "Content-Type: application/json" --data-binary @req.json
```

The endpoints, the multipart upload variant and the response fields are documented in
[API.md](API.md). The browser UI in `web/` and the Tetris demo in `tetris/` work unchanged
against this server. Only one request uses the GPU at a time (requests are serialized with a
lock), so run more processes behind a load balancer for throughput. Without `--host 0.0.0.0`
the server only listens on localhost; to reach it from elsewhere, use a tunnel such as
`ngrok http 8100`.

### Standalone merged model

To serve with other tools, or to skip the merge at every start, write the merged bf16 model
once:

```bash
python predict_jevomir.py --save-merged models/jevomir-v2          # ~8.5 GB
python api_server.py --model models/jevomir-v2 --probe artifacts/jevomir-v2/head
```

The saved model is a normal Hugging Face `AutoModelForImageTextToText` checkpoint. It is
bit-identical to the model the head was trained on. To get the same probabilities in other
tools, use this repo's prompt: `prompt_text()` and `prepare()` in `extract_cauldron.py`, with
images letterboxed to 448×448. Then read the logits of the option-letter tokens at the last
prompt position.

### Use from Python

```python
from pathlib import Path
from PIL import Image
import api_server as jev

jev.load("Qwen/Qwen3.5-4B", Path("artifacts/jevomir-v2/adapter"), Path("artifacts/jevomir-v2/head"))
result = jev.score("How many cylinders are there?", ["0", "1", "2", "3"], [Image.open("scene.png")])
print(result["prediction"]["text"], result["calibrated_p_correct"])
```

### Reproduce

Training uses one 8-GPU node through Slurm; see the comments in `run_train_8gpu.sh`,
`train_8gpu.sbatch` and `submit_pipeline.sh`. `submit_pipeline.sh` chains the SFT and RFT
stages. v2's policy stage and evaluation were then run as:

```bash
sbatch train_8gpu.sbatch grpo --out runs/v2/grpo --init-adapter runs/v2/rft/adapter-round1 --no-4bit \
    --train-subsets tallyqa:4000,nlvr2:4000,iconqa:3000,clevr:2500,vqav2:2500 \
    --skip-train-subsets tallyqa:8000,nlvr2:8000,iconqa:6000,clevr:5000,vqav2:5000 \
    --lambda-consistency 0.5 --beta-kl 0.1 --kl-ref init --epochs 1 --batch 4 --grad-accum 1 \
    --lr 3e-5 --warmup 10 --seed 0
sbatch train_8gpu.sbatch extract --adapter runs/v2/grpo/adapter --out runs/v2/head-features \
    --save-model runs/v2/final/model \
    --skip-train-subsets tallyqa:12000,nlvr2:12000,iconqa:9000,clevr:7500,vqav2:7500
python train_probe_cauldron.py --features runs/v2/head-features --out runs/v2/head --holdout-subsets --epochs 300
torchrun --nproc_per_node 8 eval_compare.py --out runs/v2/compare --baseline rft \
    --model-spec base=- rft=runs/v2/rft/adapter-round1 new=runs/v2/grpo/adapter
```

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
[API.md](API.md). The trained probe (probe-002) is committed in `artifacts/cauldron`, so no
training is needed; the model (Qwen3.5-4B @ `851bf6e`) is downloaded from Hugging Face on first
start. It needs a CUDA GPU that holds the 4B model in bf16 (developed on an H100).

```bash
uv venv .venv && uv pip install -r requirements.txt
uv pip install fastapi uvicorn python-multipart    # API-only dependencies
uv pip install flash-linear-attention              # optional: fast Qwen3.5 kernels (needs python3.x-dev)
export JEVOMIR_API_KEY=$(python -c "import secrets; print('jev_' + secrets.token_urlsafe(32))")
echo "$JEVOMIR_API_KEY"                             # the key clients send
python api_server.py --probe artifacts/cauldron     # http://127.0.0.1:8100
```

To reach it from other machines, expose the port with a tunnel such as `ngrok http 8100`.
A free ngrok tunnel drops connections above about 100 requests a minute.

`web/` is a browser UI for the API (image upload, question, options, results); run
`python web/server.py` and see [web/README.md](web/README.md) for how it integrates.

`tetris/` lets the model play Tetris live through the API; see [tetris/README.md](tetris/README.md).
