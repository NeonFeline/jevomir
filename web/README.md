# jevomir web UI

A single-page UI for the [scoring API](../API.md): upload one or two images, ask a
closed-choice question, list the options, and see the prediction, the probability of every
option and the probe's calibrated probability that the prediction is correct.

Two files, no build step and no dependencies beyond the Python standard library:

- `index.html` - the page (vanilla JS/CSS, interface in Polish)
- `server.py` - serves the page and proxies API calls

## Run

```bash
echo 'jev_...' > web/.api-key                              # git-ignored; or export JEVOMIR_API_KEY=jev_...
python web/server.py                                       # API on this machine (api_server.py, port 8100)
python web/server.py --api-url https://....ngrok-free.dev  # or a remote API (also: JEVOMIR_API_URL)
```

Open <http://127.0.0.1:8080>. Options: `--port`, `--host`, `--api-url`, `--key-file`.
Do not open `index.html` as a file; it has to be served by `server.py` (the page says so if
you do).

## How it integrates with the API

```
browser ──/api/info───▶ web/server.py ──GET /v1/info──▶ api_server.py (GPU)
        ──/api/score──▶  + X-API-Key   ──POST /v1/score─▶  (local or via ngrok)
```

**Why a proxy.** `api_server.py` sends no CORS headers, so a page on another origin cannot
call it from the browser. More importantly, the key would have to be shipped to the
browser, where anyone who opens the page can read it. `server.py` keeps the key on the
server: the browser talks only to `/api/*`, and the proxy adds `X-API-Key` (plus
`ngrok-skip-browser-warning`, so ngrok's interstitial page does not replace the JSON).

**Routes.** Only two paths are forwarded; everything else is a 404:

| Page calls | Proxy forwards to | Used for |
|---|---|---|
| `GET /api/info` | `GET /v1/info` | model name and probe run shown in the header |
| `POST /api/score` | `POST /v1/score` | scoring |

Status codes and response bodies are passed through unchanged, so API errors (401, 413,
422 with FastAPI's `detail` list) are shown in the page as returned. If the API cannot be
reached, the proxy answers `502` with `{"detail": "API unreachable: ..."}`. Request bodies are
capped at 30 MB (two 10 MB images after base64 encoding).

**Request built by the page.** Images are read in the browser and sent as `data:` URLs in
the JSON body of `/v1/score`, which the API accepts as-is:

```json
{
  "question": "Is the image red?",
  "options": ["Yes: The whole image is a solid red color", "No", "Cannot tell: The image does not show enough to decide"],
  "images": ["data:image/png;base64,iVBORw0..."]
}
```

**Option descriptions.** Each option in the form has a short label and an optional
description. The API takes plain strings, so the page sends `label: description` (or just
the label) as the option text; the model sees the description and can use it. The combined
text must fit the API's 300-character limit, which the page checks before sending. The
result maps options back by index and shows label and description separately.

**Result.** The page reads `prediction.index`, `options[].probability`, `raw_confidence`,
`calibrated_p_correct`, `input_tokens` and `timing`, and shows the full JSON under
"Surowa odpowiedź JSON". See [API.md](../API.md#what-the-probabilities-mean) for what the
numbers mean; in particular only `calibrated_p_correct` is calibrated.

## Limits and notes

- Limits are the API's: 1-2 images (JPEG/PNG/WebP, 10 MB each), a question of up to 2000
  characters, 2-16 unique options. The page checks the image count and size and the option
  count before sending; the API enforces the rest.
- The API serves one request at a time, so parallel users queue. The first request with a
  new input length can take about a second while kernels compile.
- `server.py` binds to `127.0.0.1` by default. With `--host 0.0.0.0` everyone who can reach
  the port can use the GPU through it without knowing the key; put authentication in front
  of it before exposing it.
