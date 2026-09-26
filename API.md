# jevomir scoring API

Ask a closed-choice question about one or two images. The API scores every option in one
forward pass of Qwen3.5-4B (no text generation) and returns:

- `probabilities` for each option (softmax over the option letters, uncalibrated),
- `prediction`, the most likely option,
- `calibrated_p_correct`, a probe's estimate that the prediction is correct.

Interactive schema: `GET {BASE_URL}/docs`. Server code: `api_server.py`.

## Authentication

Every `/v1/*` call needs the API key, in either header:

```
Authorization: Bearer <API_KEY>
X-API-Key: <API_KEY>
```

A missing or wrong key returns `401`. `/health` and `/docs` need no key.

## Endpoints

### `GET /health`

`{"status": "ok"}` once the model is loaded (`"loading"` before).

### `GET /v1/info`

Model and probe identity plus input limits.

### `POST /v1/score` (JSON)

| Field | Type | Required | Notes |
|---|---|---|---|
| `question` | string | yes | 1-2000 characters |
| `options` | array of strings | yes | 2-16 unique options, each 1-300 characters |
| `images` | array of strings | yes | 1-2 images as base64 (JPEG/PNG/WebP), raw or `data:` URL; at most 10 MB each |
| `id` | string | no | echoed back |

Use two images only for questions about a pair (e.g. "the left image shows two dogs").

```bash
printf '{"id":"q1","question":"How many cylinders are there?","options":["0","1","2","3"],"images":["%s"]}' \
  "$(base64 -w0 scene.png)" > req.json
curl -X POST "$BASE_URL/v1/score" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  --data-binary @req.json
```

Send large bodies from a file (`--data-binary @req.json`); a base64 image inline on the
command line can exceed the shell's argument length limit.

### `POST /v1/score/upload` (multipart)

Same scoring with file upload. Repeat `options` once per option, in order; repeat `images`
for a second image.

```bash
curl -X POST "$BASE_URL/v1/score/upload" -H "X-API-Key: $API_KEY" \
  -F "question=Is there a metal sphere?" -F options=Yes -F options=No \
  -F images=@scene.png
```

### Python

```python
import base64, requests

BASE_URL, API_KEY = "https://....ngrok-free.dev", "jev_..."
image = base64.b64encode(open("scene.png", "rb").read()).decode()
r = requests.post(f"{BASE_URL}/v1/score", headers={"Authorization": f"Bearer {API_KEY}"},
                  json={"question": "What color is the large sphere?",
                        "options": ["Gray", "Red", "Blue", "Brown"], "images": [image]},
                  timeout=60)
r.raise_for_status()
result = r.json()
print(result["prediction"]["text"], result["calibrated_p_correct"])
```

## Response

```json
{
  "id": "q1",
  "options": [
    {"letter": "A", "text": "0", "probability": 0.0012},
    {"letter": "B", "text": "1", "probability": 0.0019},
    {"letter": "C", "text": "2", "probability": 0.9894},
    {"letter": "D", "text": "3", "probability": 0.0076}
  ],
  "prediction": {"index": 2, "letter": "C", "text": "2"},
  "raw_confidence": 0.9894,
  "calibrated_p_correct": 0.999,
  "input_tokens": 249,
  "timing": {"preprocess_s": 0.006, "forward_s": 0.066, "total_s": 0.073},
  "model": {"source": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "...": "..."},
  "probe": {"run": "probe-002"},
  "notes": ["..."]
}
```

- `raw_confidence` is the `probability` of the predicted option.
- `calibrated_p_correct` is the probe's probability that `prediction` is correct, capped to
  [0.001, 0.999]. It says nothing about the other options.

## Errors

| Status | Meaning |
|---|---|
| 401 | missing or invalid API key |
| 413 | image larger than 10 MB |
| 422 | invalid input: wrong option count, duplicate or empty options, bad base64, undecodable image |

Requests are processed one at a time on a single GPU; concurrent requests queue.

## What the probabilities mean

- Options are shown to the model as letters A-P in the order given. Order can change the
  result; for important decisions, also score a reordered copy.
- Images are letterboxed to 448x448 (aspect ratio kept, gray padding). Very small details
  may be lost.
- `probabilities` are conditional on the listed options and are not calibrated.
- `calibrated_p_correct` comes from a probe on the model's hidden states, trained on 237k
  Cauldron questions (CLEVR, VQAv2, COCO-QA, A-OKVQA, Visual7W, TallyQA, NLVR2, IconQA).
  On held-out images from those sources: ECE 0.010, AUROC 0.932. On an unseen source
  (VSR spatial relations): ECE 0.048, AUROC 0.771. Expect it to be less reliable for other
  image types (documents, charts, screenshots), questions that require reading text in the
  image, and question styles far from these datasets.
- 0.999 on easy synthetic scenes is common; it is not a guarantee.
