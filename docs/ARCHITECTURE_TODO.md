# Architecture To-Do

Changes needed for the code to match the SRS and the SDD architecture. Check items off as they're done.

## 1. Extension permissions and server address

**Files:** `extension/public/manifest.json`, `extension/src/api.ts`
**SRS:** 5.3.1.41 (permission scope), 5.5.2.2 (HTTPS for all API communication)

| | Now | Target |
|---|---|---|
| Required permissions | `activeTab`, `scripting` | `activeTab`, `scripting`, `storage`, `clipboardWrite` |
| Server host permission | `http://127.0.0.1:8000/*` | The RealEye HTTPS service, e.g. `https://<realeye-host>/*` |
| Image origins | None | HTTP and HTTPS image origins as **optional** host permissions only, requested when the user clicks "Allow image access" |
| `API_URL` in `api.ts` | `http://127.0.0.1:8000` | The HTTPS service address for releases. Keep localhost only for development, e.g. chosen with a build-time setting. |

- [ ] Add `storage` and `clipboardWrite` to `permissions`
- [ ] Add the HTTPS server host to `host_permissions`
- [ ] Add `optional_host_permissions` for `http://*/*` and `https://*/*` (image origins)
- [ ] Make the API address configurable: localhost for development, HTTPS for releases

## 2. Text extraction rules

**File:** `extension/src/extract.ts`
**SRS:** 5.3.2.28, 5.3.2.29, 5.3.2.32, 5.3.2.33, 5.3.2.34

| | Now | Target |
|---|---|---|
| Cap | 10 text blocks (`MAX_BLOCKS = 10`) | First 100 eligible text blocks per scan (5.3.2.32) |
| Batching | One `/analyze` request | Split into requests of at most 10 items each (5.3.2.34) |
| Elements | `article`, `main`, `[role='main']`, `section`, `p`, `li`, `blockquote` | Visible `p`, `li`, `h1`–`h6`, `blockquote`, `pre`, `figcaption`, without repeating descendant text already assigned to an item (5.3.2.29) |
| Too short or too long text (outside 200–5000 characters) | Silently dropped | Marked as **skipped**, not truncated (5.3.2.33) |
| Over the cap | No notice | Show a scan-limit notice when more eligible items exist than the cap (5.3.2.28) |

- [ ] Raise the cap to 100 and send items in batches of 10 or fewer with the same `scan_id`
- [ ] Use the SRS element list for forming text blocks
- [ ] Return skipped items (with the reason) instead of dropping them, and show them in the popup
- [ ] Show a scan-limit notice when the cap is reached
- [ ] Update `extract.test.ts` for the new rules

## 3. Pin Hugging Face model and dataset revisions

**Files:** `ml/text/mage.py`, `ml/text/train_text.py`
**SRS:** 5.3.4.7 (base model record), 5.3.4.11 (dataset and model version), 5.5.1.6 (exact model revisions)

The SDD (6.1.2.31) records these revisions. The code now loads them by commit, so a future change on Hugging Face cannot silently change training. The October 2026 training run predates this change and did not record its revision.

| | Revision (commit) |
|---|---|
| `distilbert/distilroberta-base` | `fb53ab8802853c8e4fbdbcd0529f21fc6f459b2b` |
| `yaful/MAGE` | `342663f0a2b775455c023f5d36a1341ff0ec5402` |

- [x] Pass `revision=` for MAGE in `load_dataset("yaful/MAGE", ...)` (`mage.py`)
- [x] Pass `revision=` to `AutoTokenizer.from_pretrained` and `AutoModelForSequenceClassification.from_pretrained` for the base model (`train_text.py`)
- [x] Record both revisions in `manifest.json` and `metrics.json`

## 4. Save the confusion matrix in evaluation results

**Files:** `ml/text/calibrate.py` (the `metrics` function), `ml/text/evaluate_test.py`
**SRS:** 5.3.4.9 (precision, recall, F1, and a confusion matrix on held-out test data), 5.3.4.10 (class counts)

`metrics()` computes the true/false positive and negative counts but only returns the rates, so `test_results.json` has no confusion matrix.

- [ ] Return `tp`, `fp`, `tn`, `fn` from `metrics()` alongside the rates
- [ ] Print the confusion matrix (actual human/AI × predicted human/AI) in `evaluate_test.py` and save it in `test_results.json`
- [ ] For the current release, recompute it from the saved `test_scores.csv` at the saved cutoff (0.8118), without rerunning the model, so the test set is not used again
