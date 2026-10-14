# RealEye — Progress

*Last updated: October 13, 2026*

## Build plan

| Stage | What | Status |
|---|---|---|
| 0 | Repo setup: `extension/`, `server/`, `ml/` | ✅ Done |
| 1 | Agree on the API contract (`docs/api-contract.md`) | ✅ Done |
| 2 | End-to-end skeleton with a **fake** model: popup → page text → server → results in popup | ✅ Done (tested Oct 13) |
| 3 | Three parallel tracks (see below) | ⏭ Next |
| 4 | Replace the fake model with the real trained models | |
| 5 | Share links (SQLite, 7-day expiry) | |
| 6 | Testing: accessibility, keyboard, timing targets (SRS 5.4), held-out model evaluation | |
| 7 | Deploy the server on an HTTPS Linux host; package the extension | |

### Stage 3 tracks (can run in parallel thanks to the contract)

| Track | Work |
|---|---|
| **A. Extension + page** | Image extraction, scans over 10 items (multiple requests), progress + cancel, results panel, inline warning labels on the page, help view |
| **B. Server** | Input limits (text length/language, image size/format), 413/403/429 errors, the warning rule as a tested function, C2PA check |
| **C. ML** | Collect + license datasets, train/validation/test splits, fine-tune text + image models, pick thresholds on validation data, measure F1 ≥ 0.75 and false-positive rate ≤ 10% |

## Decisions so far

- **Stack** (from SRS 5.5.1): Chrome Manifest V3 + React 18 + TypeScript; FastAPI on Python 3.11; PyTorch + Hugging Face Transformers.
- **API contract** — `POST /analyze`, up to 10 items per request:
  - Images are sent as **base64 inside the JSON** (simpler; multipart can replace it later without changing other fields).
  - `item_id` is a **UUID** created by the extension.
  - More than 10 items → **HTTP 422** (request is readable but breaks a rule). Empty request → 400.
- **SynthID:** plan for `"unavailable"` from day one — Google may not offer a detector API we're allowed to use. The SRS allows this.

## What exists now

**`server/`**
- `app/schemas.py` — the request/response format as Pydantic models (FastAPI uses these to reject bad requests automatically).
- `app/main.py` — `GET /health` and `POST /analyze`. The only fake part is `fake_result()`: every item gets estimate 0.5, label "human", state "inconclusive", C2PA/SynthID "unavailable" (C2PA "not_applicable" for text). **When models are ready, only this function changes.**
- Auto-generated API test page at http://127.0.0.1:8000/docs while the server runs.

**`extension/`**
- `src/extract.ts` — runs inside the webpage; collects up to 10 `<p>`/`<li>`/`<blockquote>` blocks of 200–5000 characters, no duplicates. Text only for now.
- `src/api.ts` — TypeScript types matching the contract + `analyze()` which calls the server.
- `src/Popup.tsx` — Start scan button, "Scanning...", results list, "Service unavailable" if the server is down.
- `public/manifest.json` — permissions: `activeTab`, `scripting`, and the local server `http://127.0.0.1:8000`.

**`ml/`** — empty except a README.

## Known limitations of the skeleton

- Results are fake (always 50%).
- Text only; no images yet.
- Max 10 text blocks per scan (SRS allows 100 text + 50 images via multiple requests).
- Server URL is hard-coded to `http://127.0.0.1:8000` (local only; production must be HTTPS).
- No inline warnings on the page yet — results only show in the popup.
- No rate limiting, origin check, or size limits on the server yet.

## Note

An unrelated Chrome Web Store extension named **"RealEyesAI"** exists. It is not ours — worth knowing because of the similar name, and possibly worth citing as related work.

*Prepared with help from Claude (Anthropic's AI assistant).*
