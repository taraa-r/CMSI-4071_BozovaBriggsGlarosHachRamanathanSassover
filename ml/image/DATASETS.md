# Image Model — Base Models and Datasets

Source and license record for the image classifier (SRS 5.3.4.3, 5.3.4.7).
Owners: Noa, Tara. Decided: November 9, 2026.

## Label convention

**0 = real (human-captured or human-made), 1 = AI-generated**, the same as the text model. OpenFake labels images with the strings `"real"` and `"fake"`: map `"real"` → 0 and `"fake"` → 1 on load.

## Base models

| Role | Model | License | Revision (Hugging Face commit) | How it's trained | Why |
|---|---|---|---|---|---|
| **Primary** | [`openai/clip-vit-base-patch16`](https://huggingface.co/openai/clip-vit-base-patch16) | MIT ([openai/CLIP](https://github.com/openai/CLIP/blob/main/LICENSE)) | `57c216476eefef5ab752ec549e440a49ae4ae5f3` | Image encoder frozen; only a small classification layer is trained on its features (see CLIP training) | ViT design pretrained on ~400M image–caption pairs from the internet. CLIP-based detectors are reported to generalize better to generators not seen in training. Features are computed once, so training is cheap. |
| Comparison | [`google/efficientnet-b0`](https://huggingface.co/google/efficientnet-b0) | Apache-2.0 | `3d8d75d0812b65064f0151b456f869c6344bf20c` | Fully fine-tuned | Small, fast CNN pretrained on ImageNet. Works differently from CLIP (sliding filters, fine pixel traces), which makes it a useful contrast. Expected to be weaker on unseen generators and on compressed web images — the comparison tests that. |

Keep whichever does better on **validation** data. Unseen generators are measured only at the end, on `core/test` (see below).

Considered and not chosen: [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k) (Apache-2.0) — the same ViT design as CLIP's image encoder but trained only on labeled ImageNet-21k categories; CLIP vs. EfficientNet is the more informative comparison.

## Dataset

| Role | Dataset | License | Revision (Hugging Face commit) | Size | Notes |
|---|---|---|---|---|---|
| **Train / validation / test** | [OpenFake](https://huggingface.co/datasets/ComplexDataLab/OpenFake) (`ComplexDataLab/OpenFake`, subset `core`) | CC-BY-NC-4.0 (non-commercial use only) | `00b2f0ae3f5f7684c4b00f794f4c500076a9355d` | 2.3M train (3.3 TB) · 58.5k validation (72 GB) · 91.4k test (68 GB) | Dozens of generators, including 2025–2026 models (e.g. `flux.2-dev`, `imagen-4.0`); validation alone has 58. Real images: LAION + Pexels (train/validation), DOCCI + ImageNet (test). Each AI image records its generator (`model`) and `release_date`. Too large to download: read labels and sampled chunks directly from Hugging Face (`explore_openfake.py`). **Pin the revision:** the dataset was still being updated in November 2026. |
| Extra in-the-wild test (optional) | OpenFake subset `reddit` (`test` split) | CC-BY-NC-4.0 | same | 25 GB | AI images from AI-generation subreddits, real photos from photography subreddits, with real platform compression. Use only for a final check, never for training or choosing the cutoff. |

Considered and not chosen:
- [GenImage](https://github.com/GenImage-Dataset/GenImage) (CC BY-NC-SA 4.0) — older generators (Midjourney, Stable Diffusion, ADM, GLIDE, Wukong, VQDM, BigGAN), real images only from ImageNet, distributed as very large archives via Google Drive / Baidu. Real and AI images are reported to differ in file format (a shortcut a model can learn). Possible extra test set only.
- [ELSA D3](https://huggingface.co/datasets/elsaEU/ELSA_D3) — no license listed on its page; real images are only URLs that must be downloaded from the web.

## Unseen generators: already built in

OpenFake's `core/test` split is out-of-distribution by design (dataset card): its AI images come only from generators **never used in train or validation** (e.g. `gpt-image-1.5`, `gpt-image-2.0`, `nano-banana-pro`, `flux.2-klein-9b`, `z-image-turbo`, `recraft-v2`/`v3`, `midjourney-7`, `ideogram-2.0`), and its real photos come from different sources (DOCCI + ImageNet). So the test score already measures unseen generators and unseen real photos; we don't need to hold generators out ourselves. Validation uses the same generators and real sources as training.

Do **not** add OpenFake's `supplement_*` subsets to training without filtering: some of their generators are `core/test` generators (dataset card).

## Preprocessing (to match the extension and SRS 5.5.3)

**Shortcut check (Nov 9, `explore_openfake.py`, 627 sampled validation images): real and AI images differ a lot, so a model could learn file properties instead of signs of AI.**

| | Real | AI |
|---|---|---|
| File format | 98% JPEG | 51% JPEG, 48% PNG |
| JPEG quality (median) | 83 | 95 |
| File size (median) | 105 KB | 612 KB |
| Image size (median) | 0.6 MP (LAION 0.33 MP, Pexels 23 MP) | 1.0 MP (often 1024×1024) |
| Has camera data (EXIF) | 8% | 15% |
| Within SRS image limits | 69% (only 14% of Pexels) | 99% |

So data prep must treat every image, real and AI, the same way.

## Data prep (Nov 10)

`prepare_data.py` builds the train and validation sets; `check_shortcuts.py` confirms the shortcuts are gone. Both read OpenFake straight from Hugging Face at the pinned revision; the test split is never read.

Every image goes through `preprocess.normalize_image`, the same function the server will use:

1. Apply the EXIF rotation, convert to RGB, drop all metadata.
2. **Square crop** (random 80–100% of the shorter side for train; centered and full for validation, test and the server). This removes the "1024×1024 square = AI" shortcut.
3. **Resize** to a random 224–320 px for train, fixed 288 px otherwise (CLIP and EfficientNet read 224 px). Images under 400 px on their shorter side are skipped in both groups, so nothing is upscaled. Upscaling only the small LAION photos while shrinking AI images would have created a new shortcut. Oversized Pexels photos shrink instead of being dropped.
4. **Re-encode as JPEG**, quality random 70–95 for train and fixed 90 otherwise. This removes the PNG, JPEG quality, file size and EXIF differences.

Selection:
- Skips duplicate `image_sha256`, and validation images that also appear in train.
- Excludes `tiny-random-sana`. OpenFake's own captions describe its images as "a uniform field of digital noise".
- Normalizes `type` spellings (`Base` → `base`, `Fine-tune` → `finetune`, `LoRA` → `lora`).
- Balances to the same number of real and AI images. Within each group, every generator (or real source) gets an equal share, up to what is available. With about 58 generators and 20,000 AI images, that's about 345 per generator, so a few common generators (e.g. `sd-2.1`, `flux.1-dev`) can't dominate. Real images are 50% LAION (everyday web images) and 50% Pexels (professional stock photos), instead of the natural mix of about 64/36. That way neither kind of real photo looks "unusual" to the model.
- Reads random chunks of about 70 images each (well mixed between real and AI, and across generators), 30% more than needed, so balancing can trim. Defaults: 20,000 per class for train, 2,500 for validation.

**Shortcut check after prep** (small test run: 200 + 200 train, 100 + 100 validation). The AUC column says how well that one property alone tells real from AI: 0.5 = coin flip, 1.0 = perfect shortcut.

| | Before (Nov 9) | After: real | After: AI | AUC train | AUC validation |
|---|---|---|---|---|---|
| File format | real 98% JPEG, AI 48% PNG | 100% JPEG | 100% JPEG | – | – |
| JPEG quality (median) | 83 vs 95 | 82 / 90 | 82 / 90 | 0.49 | 0.50 |
| Image size | 0.6 vs 1.0 MP | 274 / 288 px | 272 / 288 px | 0.48 | 0.50 |
| EXIF | 8% vs 15% | 0% | 0% | – | – |
| File size (median) | 105 vs 612 KB | 16.1 / 22.9 KB | 17.6 / 26.0 KB | 0.55 | **0.58** |

The remaining file-size gap is small and varies mostly by generator (e.g. 11 KB for `flux.1-dev`, 23 KB for `sd-2.1`). After identical processing, file size only measures how much fine detail a picture has. The model never sees file size, but a detail difference is in the pixels. It could be a real sign of AI (over-crisp generator output) or a side effect of where the real photos came from (softer, previously compressed web images), in which case the model could learn "sharp = AI" and flag crisp professional photos. Plan:
1. Recheck on the full run (COLAB.md, step A4). With 100 images per group, 0.58 may be chance.
2. Blur and JPEG augmentation during training makes detail level unreliable as a clue (see CLIP training).
3. After training, `train_clip.py` reports how strongly scores follow file size (Spearman correlation) and the false-positive rate for LAION vs. Pexels.
4. Only if 2 and 3 show a problem: keep equal real and AI images at each detail level, like the style cap.

## Style cap (decided Nov 11)

OpenFake's AI images include many illustrations, anime and paintings, while its real images are almost all photos. A model could learn "illustration = AI". OpenFake has no style column (`prompt_kind` only says where the prompt text came from), so `label_styles.py` asks CLIP (the same pinned revision) which description each prepared image matches best: a photo, an illustration, a 3D render, or a graphic with text.

**Decision:** AI images that CLIP is fairly sure are illustrations (best match, score ≥ 0.5) are capped at the same share as on the real side; real images are trimmed so real = AI. The model therefore mostly compares photos with photo-style AI images, which fits RealEye's main use (fake photos). Only the illustration label is acted on: in a spot check it was mostly right, while "3D render" also caught real studio photos.

Small test run (CLIP style, share within each group):

| | Real | AI before | AI after cap |
|---|---|---|---|
| Illustration, train | 2% | 8% | 2% |
| Illustration, validation | 0% | 14% | 4% |
| Photo, train | 72% | 59% | 63% |

The cap dropped 24 of 400 train and 20 of 200 validation images. Remaining "illustration" rows after the cap are ones CLIP was unsure about. `styles.parquet` keeps every image's style so errors can be checked by style after training; both CLIP and EfficientNet train and validate only on rows with `keep == True`. On the test split, report results both overall and by style.

## CLIP training (Nov 11)

**Why CLIP stays frozen and EfficientNet would be fully fine-tuned.** Both start from pretrained weights; neither is trained from scratch. CLIP learned from ~400M image–caption pairs whose captions describe style and quality ("a watercolor", "a blurry phone photo", "digital art"), so its features already carry many traits of AI images, and a small classifier on top works well (Ojha et al. 2023, "Towards Universal Fake Image Detectors"). Leaving the encoder unchanged keeps that broad knowledge, which helps with unseen generators. EfficientNet was trained only to name objects on ImageNet, so its features largely ignore the fine pixel traces that reveal AI; frozen, it would do poorly, so it has to be fully fine-tuned. EfficientNet is postponed; a frozen EfficientNet with a classifier on top could be added cheaply as an extra baseline.

**Pipeline** (after data prep and the style cap):

1. **`extract_features.py`** runs the frozen CLIP image encoder once per image version and saves 512 numbers per version (`get_image_features`). Train images get **3 versions**: the clean image plus 2 augmented ones. Validation gets only the clean image.
2. **`train_clip.py`** trains one linear layer (512 numbers in, one score out) on those features.

**Blur and JPEG augmentation** (train only, same for real and AI). Each augmented version gets, independently:
- a Gaussian blur with probability 0.5 (strength σ 0.1–3.0),
- an extra JPEG compression with probability 0.5 (quality 30–95).

This is the recipe from Wang et al. 2020, "CNN-generated images are surprisingly easy to spot". Each training round picks one random version of every image, so the classifier can't rely on sharpness or compression level (the file-size gap above), and it holds up better on images that social media has recompressed. Augmentations are seeded per image, so reruns give the same features.

**Classifier training:**
- Features are L2-normalized, then standardized (each of the 512 numbers minus its train mean, divided by its spread). Without standardization the numbers are tiny (~0.04) and the layer learns very slowly. The mean and spread are saved with the layer, so the server applies the same step.
- AdamW, learning rate 0.001, batches of 256, 30 epochs (about 145 steps per epoch on the full train set).
- Four weight-decay settings are tried (0, 0.01, 0.1, 1). For each, the epoch with the **lowest validation loss** is kept, then the best setting overall. Validation loss is used instead of accuracy because it also rewards well-judged confidence, which matters for calibration. The script warns if the best epoch is the last one (needs more epochs).
- The 0.5 cutoff in its printout is a first look only; the real cutoff is chosen in calibration, as for the text model.

**What it reports** (`metrics.json`, `validation_scores.csv`): F1, accuracy, false-positive rate, recall, precision and AUC, plus the counts; the false-positive rate by real source (LAION / Pexels) and by style; recall by style and by generator; and how strongly scores follow file size.

**Splits:** train teaches the model; validation is used during development (choosing settings, comparing models, calibration and the cutoff, data checks); `core/test` is used once, at the very end.

Small test run (376 train images): the pipeline runs end to end. Scores at that size mean nothing; real results come from the full run.

## Full run

See [COLAB.md](COLAB.md):
- **Part A** (CPU runtime): prepares the data and rechecks the shortcuts. It resumes after a disconnect.
- **Part B** (GPU): labels styles, extracts CLIP features with blur/JPEG augmentation, and trains the classifier.

**Next steps after the full run:** calibration and the cutoff on validation (Platt scaling, as for the text model), then the one-time test on `core/test` (overall, by style, and on unseen generators), then server integration with the same `normalize_image`.

## Known limitations

- CC-BY-NC-4.0 allows non-commercial use only; a commercial RealEye would need different training data.
- OpenAI's model card says CLIP was built for research and not for general deployment without careful study of the specific use.
- Drawings, anime and digital art are not reliably judged: AI illustrations are capped in training and real illustrations are rare in OpenFake (see Style cap). The model is meant for photos and photo-style images.
- New generators appear constantly; results describe OpenFake's test data, not every image on the web (SRS 5.5.3).
