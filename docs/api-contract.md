# RealEye API Contract

This is the agreed shape of data between the Chrome extension and the server.
Once the team signs off, the extension, server, and ML work can proceed in parallel against it.
Section numbers (e.g. 5.3.3.16) point to the SRS.

## `POST /analyze`

One request = one batch of up to **10 items** (5.3.3.69), body at most **16 MiB** (5.3.3.68).
A full-page scan with more items sends several requests with the same `scan_id`.

### Request

```json
{
  "scan_id": "b3f1c2e0-6a4d-4f7e-9c1a-2d8e5f0a7b11",
  "page_url": "https://example.com/listing/123",
  "scan_time": "2026-10-13T18:42:05Z",
  "items": [
    {
      "item_id": "4c7d2a91-0e3b-4f6a-8b5d-91e2c7f0a3d4",
      "type": "text",
      "text": "Luxury beachfront apartment, only $800/month. Pay the deposit today to secure..."
    },
    {
      "item_id": "9a1e5f37-2b8c-4d0e-a6f1-3c5b7d9e2f80",
      "type": "image",
      "image_base64": "/9j/4AAQSkZJRgABAQ...",
      "mime_type": "image/jpeg",
      "source_url": "https://example.com/img/apartment.jpg"
    }
  ]
}
```

| Field | Notes |
|---|---|
| `scan_id` | Created by the extension (UUID) when the user clicks "Start scan". |
| `item_id` | Created by the extension as a UUID (decided). Server echoes it back (5.3.3.4, 5.3.3.15). |
| `type` | `"text"` or `"image"` (5.3.3.2). |
| `text` | 200–5000 characters after whitespace normalization, English (5.5.3). |
| `image_base64` | Base64 of the **original, unmodified bytes** of the image (decided: base64 in JSON, not multipart) so C2PA can be checked (5.3.3.7). JPEG, PNG, static WebP; ≤ 10 MiB. |

### Response — `200 OK`

```json
{
  "scan_id": "b3f1c2e0-6a4d-4f7e-9c1a-2d8e5f0a7b11",
  "rule_version": "rule-1.0",
  "results": [
    {
      "item_id": "4c7d2a91-0e3b-4f6a-8b5d-91e2c7f0a3d4",
      "type": "text",
      "overall_state": "warning",
      "conflict": false,
      "warning_evidence": ["classifier"],
      "classifier": {
        "status": "completed",
        "estimate": 0.87,
        "estimate_kind": "model_estimate",
        "label": "ai",
        "threshold": 0.8118,
        "model_version": "distilroberta-base-20261030-0247",
        "reason": null,
        "analyzed_chars": 2040,
        "total_chars": 4800
      },
      "c2pa": { "status": "not_applicable" },
      "synthid": { "status": "unavailable" },
      "limitations": ["Not all AI-generated content contains detectable credentials or watermarks."],
      "explanation": "The model estimate (87%) is at or above the AI cutoff (81%). This estimate is based on the first 2,040 of 4,800 characters."
    },
    {
      "item_id": "9a1e5f37-2b8c-4d0e-a6f1-3c5b7d9e2f80",
      "type": "image",
      "overall_state": "inconclusive",
      "conflict": false,
      "warning_evidence": [],
      "classifier": {
        "status": "completed",
        "estimate": 0.22,
        "estimate_kind": "model_estimate",
        "label": "human",
        "threshold": 0.65,
        "model_version": "image-0.1"
      },
      "c2pa": { "status": "missing_credential" },
      "synthid": { "status": "unavailable" },
      "limitations": ["Not all AI-generated content carries credentials or watermarks."]
    }
  ]
}
```

### Allowed values

**`overall_state`** (exactly one per item, 5.3.3.55 — precedence from SRS 5.1.1):
- `warning` — any warning trigger: estimate ≥ threshold, C2PA verified AI assertion, or SynthID detected
- `inconclusive` — at least one check completed, no trigger
- `unable_to_analyze` — no check completed

**`classifier.status`**: `completed` · `unsupported` (text too short/long or not English, 5.3.3.70–71 — no `estimate` returned) · `error` (model not loaded or inference failed)

**`classifier.reason`** (only when `unsupported`): `length` (outside 200–5000 characters after whitespace is collapsed) · `language` (not English) · `not_available` (image model not released yet)

**`classifier.estimate`** is the **calibrated** model estimate (Platt scaling), 0–1. **`threshold`** is the released cutoff on that same scale (0.8118 for `distilroberta-base-20261030-0247`); `estimate >= threshold` triggers a warning.

**`classifier.analyzed_chars` / `total_chars`**: the model reads at most the first 512 tokens. `analyzed_chars` is where the last token read ends in the text; when it is less than `total_chars`, the estimate covers only that first part, and `explanation` says so.

**`explanation`**: a plain-language sentence or two explaining the overall state (5.3.3.56), safe to show to users as-is.

**`c2pa.status`**: `verified_ai_assertion` · `verified_no_ai_assertion` · `missing_credential` · `invalid_credential` · `not_applicable` (text) · `unavailable` · `error`

**`synthid.status`**: `detected` · `not_detected` · `not_applicable` · `unavailable` · `error`

**`warning_evidence`**: list of which methods triggered the warning — any of `classifier`, `c2pa`, `synthid` (5.3.1.28).

**`conflict`**: `true` when there is a warning trigger *and* a below-threshold classifier estimate. It never changes `overall_state` (5.3.3.54).

### Errors

| HTTP | When | SRS |
|---|---|---|
| `400` | No supported items, or malformed request | 5.3.3.3 |
| `403` | Browser `Origin` is not the RealEye extension | 5.3.3.74 |
| `413` | Body over 16 MiB | 5.3.3.68 |
| `422` | More than 10 items, or an image outside format/size limits | 5.3.3.69, 5.3.3.72 |
| `429` | Over 60 requests/minute from one IP; includes `Retry-After` header | 5.3.3.73 |

Error body:

```json
{ "error": { "code": "too_many_items", "message": "A request may contain at most 10 items." } }
```

## `GET /health`

Returns `{"status": "ok", "model_loaded": true, "model_version": "distilroberta-base-20261030-0247", "rule_version": "rule-1.0"}`. `model_loaded` is `false` when the model folder is missing (see README); the server still answers, and text items come back `unable_to_analyze`.

## Not in this draft yet

Share links (`POST /reports`, `GET /r/{id}`) — Stage 5, will be added later.

## Decided

- Images are sent as base64 inside the JSON (simpler; ~33% size overhead counts toward 16 MiB). Multipart can replace it later without changing other fields.
- `item_id` is a UUID generated by the extension.
- More than 10 items returns `422` (request is readable but breaks a rule).
