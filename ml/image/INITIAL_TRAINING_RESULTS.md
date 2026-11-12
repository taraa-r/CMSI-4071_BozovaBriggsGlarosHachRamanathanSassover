# Image Classifier — Initial Training Results

## Overview

This is the first full training run of RealEye's image classifier, done on Google Colab on Nov 12, 2026, following `COLAB.md`. It uses **frozen CLIP ViT-B/16** features with one trained linear layer on top. The model labels an image as real (0) or AI-generated (1).

It sets an initial baseline. All results below come from the **validation set**. The held-out test set has not been used yet.

Model folder (Google Drive): `RealEye/models/image/clip-linear-20261112-0823/` (`head.pt`, `config.json`, `metrics.json`, `validation_scores.csv`).

## Data

Source: OpenFake `core`, revision `00b2f0ae3f5f7684c4b00f794f4c500076a9355d`. See `DATASETS.md` for how the data was prepared.

| Step | Train | Validation |
| --- | --- | --- |
| Prepared (`prepare_data.py`) | 20,000 real + 20,000 AI | 2,500 real + 2,500 AI |
| After the illustration cap (`label_styles.py`) | 37,808 images | 4,454 images |
| Versions per image (`extract_features.py`) | 3 (1 clean + 2 with random blur/JPEG) | 1 (clean) |

The illustration cap removed about 5% of train and 11% of validation. Validation lost more because AI illustrations were more common there (14% before the cap).

## Shortcut check on the full data

Output of `check_shortcuts.py` on the full prepared data. For each property, the AUC says how well that property *alone* tells real from AI (0.5 = coin flip, 1.0 = perfect shortcut). Values from 0.45 to 0.55 count as "ok".

| Property | Train real | Train AI | Validation real | Validation AI |
| --- | --- | --- | --- | --- |
| JPEG (share) | 100% | 100% | 100% | 100% |
| Median width (px) | 273 | 272 | 288 | 288 |
| Median file size (KB) | 16.7 | 17.5 | 24.5 | 26.3 |
| Median JPEG quality | 82 | 82 | 90 | 90 |
| Has camera data (EXIF) | 0% | 0% | 0% | 0% |
| Square | 100% | 100% | 100% | 100% |
| Within SRS image limits | 100% | 100% | 100% | 100% |

| AUC | Train | Validation | Small test run (validation) |
| --- | --- | --- | --- |
| File size | 0.533 ok | **0.568, shortcut left** | 0.58 |
| Width | 0.501 ok | 0.500 ok | — |
| Height | 0.501 ok | 0.500 ok | — |
| JPEG quality | 0.500 ok | 0.500 ok | — |

Train and validation have different medians (273 vs. 288 px, quality 82 vs. 90) on purpose. Train uses random sizes and qualities; validation, test and the server use fixed ones. Within each split, real and AI match.

**The file-size gap is real but small.** The full run confirms what the small run found: AI images compress slightly larger (26.3 vs. 24.5 KB in validation), because they have a bit more fine detail. The model never sees file size, but it does see that detail in the pixels. The training results below show that the model did **not** learn to follow it (score vs. file size Spearman correlation near 0), so no further fix is needed for now.

## Training configuration

| Setting | Value |
| --- | --- |
| Encoder | `openai/clip-vit-base-patch16` (frozen), revision `57c21647…` |
| Classifier | one linear layer, 512 → 1, on L2-normalized and standardized features |
| Augmentation (train) | blur (p = 0.5, σ 0.1–3.0) and/or JPEG (p = 0.5, quality 30–95); one random version per image each round |
| Optimizer | AdamW, learning rate 1e-3, batch size 256 |
| Rounds (epochs) | up to 30; the round with the lowest validation loss is kept |
| Weight decay tried | 0, 0.01, 0.1, 1.0 |
| Seed | 42 |
| Hardware | Google Colab T4 GPU |
| Classification cutoff | 0.5 (first look only; the real cutoff comes from calibration) |

**Weight decay** pulls the layer's weights toward 0 a little on every step. This stops the model from relying on quirks of the training images. More pull means a simpler model, but too much stops it from learning.

| Weight decay | Best validation loss | At round |
| --- | --- | --- |
| **0.0 (chosen)** | **0.3928** | 19 |
| 0.01 | 0.3929 | 19 |
| 0.1 | 0.3938 | 19 |
| 1.0 | 0.4049 | 19 |

The settings from 0 to 0.1 are practically the same, and 1.0 is too strong. A single layer trained on 37,808 images × 3 versions has little room to memorize, and the blur/JPEG versions already guard against that, so extra pull didn't help. The best round was 19 of 30, so training had settled and more rounds aren't needed.

## Initial validation results

At the 0.5 first-look cutoff:

| Metric | Result | Target (SRS) |
| --- | --- | --- |
| F1 score | 0.828 | ≥ 0.75 (5.4.8): **met** |
| Accuracy | 82.3% | — |
| False-positive rate | 20.3% | ≤ 10% (5.4.20): **not met at 0.5** |
| AUC | 0.906 | — |

Precision, recall and the confusion counts are in `metrics.json`.

### Explanation of results

**F1 (0.828)** balances catching AI images (recall) against being right when the model says "AI" (precision). It is above the 0.75 target.

**Accuracy (82.3%)** is the share of validation images labeled correctly.

**False-positive rate (20.3%)** is the share of real images wrongly flagged as AI. At the 0.5 cutoff this is twice the target. Because 0.5 is only a placeholder, this doesn't fail the model yet: raising the cutoff lowers false positives, at the cost of missing some AI images. Calibration will choose the cutoff that keeps false positives at or below 10% and show what that costs in F1.

**AUC (0.906)**: if you pick one random AI image and one random real image, the model gives the AI image the higher score about 91% of the time. AUC doesn't depend on the cutoff, so it shows that the model separates the two groups well. That is why a stricter cutoff should be able to bring the false-positive rate down without losing too much.

### Where the errors are

**False-positive rate by real source** (cutoff 0.5):

| Real source | False-positive rate |
| --- | --- |
| LAION | 28.6% |
| Pexels | 12.2% |

Most false alarms are LAION images. LAION is scraped from the web and includes graphics, edited images, low-quality photos and possibly some AI images labeled as real. Pexels is clean stock photography. So the model is more likely to flag "web-looking" real images than professional photos. This is a known weakness to check again after calibration and on the test set, whose real images come from different sources (DOCCI and ImageNet).

**Score vs. file size** (Spearman correlation; 0 means no link):

| Group | Correlation |
| --- | --- |
| Real | −0.022 |
| AI | 0.142 |

Both are near 0. The model's scores don't simply follow how detailed (and so how large) an image is, so the file-size gap from the shortcut check did not turn into a shortcut in the model. A value above about 0.3 for real images would have meant that sharp real photos tend to get flagged.

## Next steps

1. **Calibration** (`image/calibrate.py`, to be written, modeled on `text/calibrate.py`):
   - Platt scaling, so the percentage shown in the extension is honest.
   - Choose the cutoff that keeps the false-positive rate at or below 10%.
   - Report F1 at that cutoff, overall and for LAION vs. Pexels.
   - Save `calibration.json` for the server.
2. **One-time test set evaluation:** the test set includes generators and real sources that were not used in training.
3. **Server integration:** load CLIP and `head.pt`, apply `preprocess.normalize_image` with fixed settings, then calibration.
4. **Later, optional:** an EfficientNet-B0 comparison.
