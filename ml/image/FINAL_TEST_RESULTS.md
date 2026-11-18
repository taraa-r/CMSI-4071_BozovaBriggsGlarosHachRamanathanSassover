# Image Classifier — Final Results on the Held-Out Test Set

The one-time evaluation of the released image model on OpenFake's held-out `core/test` split (Nov 18). These are the official image-model results for the report. Nothing was fitted, tuned or changed after seeing them.

**Result: both SRS targets pass.** F1 **0.847** (target ≥ 0.75, SRS 5.4.9) and false-positive rate **7.2%** (target ≤ 10%, SRS 5.4.21), on 20 AI generators and two real-photo sources the model had never seen. The drop from validation is small (F1 0.862 → 0.847, recall 81.7% → 78.8%).

**Main limitation:** the 7.2% false-positive rate is an average of two very different real-photo sources. Clean, modern, high-resolution photos (DOCCI) are flagged **15.2%** of the time; older, lower-resolution web photos (ImageNet) only **0.4%**. Photorealistic and graphic-design AI images from new generators are caught only about 65–70% of the time.

## What was tested

| | |
| --- | --- |
| Model | `clip-mlp-20261113-0104`: frozen CLIP ViT-B/16 (`openai/clip-vit-base-patch16`, revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`) + two-layer head (512 → 256, GELU, dropout 0.2 → 1), weight decay 1.0, round 13 |
| Calibration | Platt scaling, a = 0.771, b = −0.359 (chosen over temperature scaling on validation; see [ERROR_ANALYSIS.md](ERROR_ANALYSIS.md#calibration-nov-17)) |
| Cutoff | 0.6205 on the calibrated estimate (raw logit score 0.7508), chosen for an 8% false-positive rate on validation half B |
| Test data | OpenFake `core/test` at revision `00b2f0ae3f5f7684c4b00f794f4c500076a9355d`, prepared by `prepare_data.py --splits test` |
| Size | 5,000 images: 2,500 real (DOCCI 1,148, ImageNet 1,352) + 2,500 AI (20 generators) |
| Preparation | Same as validation: centered square crop, resized to 288 px, re-saved as JPEG quality 90, no metadata; equal real and AI, an equal share per generator and real source. One difference: images down to **288 px** on the shorter side were kept (validation used 400 px), so small real photos such as ImageNet's stay in, as they would in real use. |
| Script | `evaluate_test.py` (COLAB.md, Part F), T4 GPU |

**Why this test is hard:** OpenFake's test split is out-of-distribution by design ([DATASETS.md](DATASETS.md)). Its AI images come only from generators never used in train or validation, and its real images come from DOCCI and ImageNet instead of LAION and Pexels. So it measures how the model does on new generators and new kinds of real photos, which is closer to real use than validation.

## 1. Checks

| Check | Result |
| --- | --- |
| Test images also in train or validation (same `image_sha256`) | **0** |
| Test generators also used in train or validation | **none** (all 20 unseen) |

The model had seen none of these images and none of these generators.

## 2. Overall results at the saved cutoff

| Metric | **Test** | Validation (half A) | Change | SRS target |
| --- | --- | --- | --- | --- |
| F1 | **0.847** | 0.862 | −0.015 | ≥ 0.75 (5.4.9): **PASS** |
| False-positive rate | **7.2%** | 7.9% | −0.7 | ≤ 10% (5.4.21): **PASS** |
| Recall (AI caught) | 78.8% | 81.7% | −2.9 | |
| Precision | 91.6% | 91.2% | +0.4 | |
| Accuracy | 85.8% | 86.9% | −1.1 | |
| AUC | 0.940 | 0.946 (all validation) | −0.006 | |

The test set is 50/50 real and AI, like validation, so F1 and precision are directly comparable.

**What it means:** moving to 20 unseen generators and two unseen real-photo sources cost about 3 points of recall and 0.015 of F1. The model learned signs of AI that carry over to new generators, not just the ones it was trained on. The false-positive rate came in below the 8% the cutoff was set for, but see section 3: that average depends heavily on which real photos are in the mix.

## 3. False positives: real images wrongly flagged as AI

181 of 2,500 real images (7.2%).

| Source | Style | Images | Flagged | % flagged | Share of all false alarms |
| --- | --- | --- | --- | --- | --- |
| **DOCCI** | **all** | **1,148** | **175** | **15.2%** | **97%** |
| DOCCI | graphic | 70 | 21 | 30.0% | 12% |
| DOCCI | illustration | 16 | 1 | 6.2% | 1% |
| DOCCI | photo | 843 | 122 | 14.5% | 67% |
| DOCCI | render | 219 | 31 | 14.2% | 17% |
| **ImageNet** | **all** | **1,352** | **6** | **0.4%** | **3%** |
| ImageNet | graphic | 3 | 0 | 0.0% | 0% |
| ImageNet | illustration | 1 | 0 | 0.0% | 0% |
| ImageNet | photo | 1,315 | 6 | 0.5% | 3% |
| ImageNet | render | 33 | 0 | 0.0% | 0% |

Average calibrated estimate given to real images: **DOCCI 0.29**, ImageNet 0.05.

Styles are CLIP's guess at what each image looks like (photo, illustration, render, graphic), the same labels as in development. Groups under about 100 images (all ImageNet non-photo groups, DOCCI illustration and graphic) can move many points by chance.

**What the contact sheet shows** (`test_errors/false_alarms_docci_top.png`, scores 0.92–0.99): ordinary real photos of everyday things: a park bench in snow, toy airplanes on a wooden floor, a cat lying on a bed, two cats at a computer, a tire with a football, a red chair by a fence, rows of refrigerators, a truck, a toy robot, a "Road Closed" sign, a sculpture, a mural. They are sharp, well lit, high resolution and usually show one clear, centered subject.

**Analysis:**

- **Almost all false alarms come from DOCCI (97%).** DOCCI is a Google dataset of recent, high-resolution photos taken deliberately to show one subject clearly. ImageNet photos are older web images: lower resolution, more compressed, more cluttered.
- **This is the "polish effect" from the error analysis, now stronger.** On validation, polished LAION images (publicity photos, posters) were flagged about 13% and stock photos (Pexels) about 3%. On test, the line falls between clean, modern photos and older, rougher ones. Modern AI generators produce exactly that clean, sharp, well-lit look, so the model leans "AI" whenever it sees it, even in an everyday snapshot.
- **The overall rate depends on the mix.** On DOCCI-like photos alone, the false-positive rate is 15.2%, above the 10% target; on ImageNet-like photos it is under 1%. Recent smartphone photos look more like DOCCI than ImageNet, so real users may see more false alarms than the 7.2% average suggests.
- **Style matters less than source.** Within DOCCI, photos and renders are flagged about equally (14–15%); graphics more often (30%, but only 70 images). The pattern is about image quality and look, not style.
- **Many of these mistakes are confident** (the top 16 all scored 0.92 or higher), so they could not be fixed by moving the cutoff a little.

## 4. Recall: AI images caught

1,969 of 2,500 AI images (78.8%).

| AI style | Images | Caught | **Test recall** | Validation recall (half A) |
| --- | --- | --- | --- | --- |
| illustration | 611 | 596 | **97.5%** | 90.3% |
| render | 611 | 527 | **86.3%** | 84.2% |
| graphic | 317 | 219 | **69.1%** | 82.7% |
| photo | 961 | 627 | **65.2%** | 78.3% |

By generator (all unseen; worst first). Many groups are small, so rates under about 50 images can be off by 10–20 points by chance:

| Generator | Images | Caught | Recall |
| --- | --- | --- | --- |
| recraft-v3 | 70 | 32 | 45.7% |
| halfmoon-4-4-25 | 26 | 14 | 53.8% |
| recraft-v2 | 13 | 7 | 53.8% |
| ideogram-2.0 | 20 | 11 | 55.0% |
| frames-23-1-25 | 14 | 8 | 57.1% |
| midjourney-7 | 279 | 178 | 63.8% |
| flux.2-klein-9b | 403 | 258 | 64.0% |
| aurora-20-1-25 | 18 | 12 | 66.7% |
| lumina-17-2-25 | 27 | 19 | 70.4% |
| z-image-turbo | 402 | 291 | 72.4% |
| veo-3 | 166 | 131 | 78.9% |
| ernie-image | 16 | 13 | 81.2% |
| gpt-image-2 | 35 | 30 | 85.7% |
| nano-banana-pro | 28 | 24 | 85.7% |
| ernie-image-turbo | 47 | 41 | 87.2% |
| wan-video-2.5 | 93 | 84 | 90.3% |
| seedream-v5.0 | 24 | 22 | 91.7% |
| sora-2 | 39 | 37 | 94.9% |
| gpt-image-1.5 | 378 | 360 | 95.2% |
| illustrious | 402 | 397 | 98.8% |

The generators are not evenly represented: the equal-share balancing is capped by how many usable images each generator has in OpenFake's test split, so five generators (flux.2-klein-9b, z-image-turbo, illustrious, gpt-image-1.5, midjourney-7) make up about three quarters of the AI images.

By OpenFake's generator type: base 61.0% (656 of 1,076), image 94.1% (982 of 1,044), video 87.1% (331 of 380). The test split uses different type names than train (base, finetune, lora), so these aren't comparable with development results.

**What the contact sheet shows** (`test_errors/missed_ai_top.png`, the AI images the model is surest are real, scores 0.00–0.02): a stormy landscape, a "Hello World" shop window, a vintage camera on a table, gold jewelry on white, a tree trunk, a hillside village, a sweater on a model, a hand in blue light, a beach with palm leaves, a clock, a fashion collage, an office building, sushi, a book with a tassel, a black-and-white shadow shot, a figure in smoke. Almost all look like **stock or editorial photography**.

**Analysis:**

- **Anime and illustration AI is easy (97.5%)**, and some generators are caught almost always (illustrious 99%, gpt-image-1.5 95%, sora-2 95%).
- **Photorealistic AI is the weak spot (65.2%)**, 13 points below validation. The large photoreal generators midjourney-7 (64%), flux.2-klein-9b (64%) and z-image-turbo (72%) drive this. It is the kind of AI image that matters most for scams and misinformation.
- **Graphic-design generators are hardest** (recraft-v3 46%, recraft-v2 54%, ideogram-2.0 55%), which explains the drop in graphic recall (82.7% → 69.1%). These tools make posters, logos and layouts that look like human design work.
- **The confident misses look like stock photography,** the look the model learned as "real" from Pexels in training. New generators that imitate stock photos get through.

## 5. Calibration and overconfidence on test

| | ECE | Log loss |
| --- | --- | --- |
| Test, raw | 0.052 | 0.336 |
| **Test, calibrated (Platt)** | **0.027** | **0.316** |
| Validation half B, raw | 0.047 | 0.313 |
| Validation half B, calibrated | 0.013 | 0.295 |
| Validation loss when the head was chosen (raw, all validation, weight decay 1.0, round 13) | | 0.313 |

Share actually AI in each estimate group (images in the group):

| Estimate | Raw | Calibrated |
| --- | --- | --- |
| 0–10% | 6% (1,721) | 6% (1,653) |
| 10–20% | 23% (269) | 22% (355) |
| 20–30% | 28% (196) | 29% (251) |
| 30–40% | 39% (162) | 43% (203) |
| 40–50% | 42% (130) | 42% (167) |
| 50–60% | 42% (126) | 57% (169) |
| 60–70% | 51% (137) | 65% (197) |
| 70–80% | 62% (198) | 74% (205) |
| 80–90% | 74% (248) | 81% (300) |
| 90–100% | 95% (1,813) | 98% (1,500) |

Wrong answers at the cutoff: **712 of 5,000**; 18% of them were over 90% sure and 1% over 99% sure.

**Analysis:**

- **Calibration holds up on new data.** Calibrated estimates are off by about 3 percentage points on average (ECE 0.027), and most groups land in or next to their own range (for example, 70–80% → 74% actually AI, 80–90% → 81%). The "Model estimate" shown in the extension stays meaningful on unseen generators.
- **It is a little worse than on validation,** as expected for new data: ECE 0.013 → 0.027 and log loss 0.295 → 0.316 (raw: 0.313 → 0.336). The low groups are slightly under-estimated (10–20% → 22% actually AI), partly the DOCCI false alarms and the confident misses.
- **Platt still helps on test:** it roughly halves the raw ECE (0.052 → 0.027) and lowers the log loss, so choosing it on validation was the right call.
- **Few extreme mistakes:** only 1% of wrong answers were more than 99% sure.

## 6. Conclusions

1. **The image model meets both SRS requirements on the independent test set:** F1 0.847 ≥ 0.75 (SRS 5.4.9) and false-positive rate 7.2% ≤ 10% (SRS 5.4.21). These are the numbers to report.
2. **It generalizes to unseen generators** with only a small loss (recall 81.7% → 78.8%), the main purpose of evaluating on OpenFake's out-of-distribution test split.
3. **Known limitations** (for the write-up, the user guide and the extension's wording):
   - **False positives depend strongly on the kind of real photo:** about 15% for clean, modern, high-resolution photos (DOCCI) vs. under 1% for older web photos (ImageNet). Users with recent phone photos may see more false alarms than the 7.2% average.
   - **Photorealistic AI and graphic-design AI from new generators are caught only about 65–70% of the time,** and AI images that look like stock photography often pass as real with high confidence.
   - These support showing the result as a "Model estimate" with wording like "may be AI-generated or digitally edited," next to the C2PA and SynthID evidence, rather than as a verdict.
4. **The model, calibration and cutoff stay as they are.** Changing them because of these results would make the test optimistic. Any future improvement (for example, more varied real photos in training, such as modern high-resolution snapshots) would need a new, unused evaluation set, such as OpenFake's `reddit` subset ([DATASETS.md](DATASETS.md)).

## Files

In the model folder on Google Drive (`RealEye/models/image/clip-mlp-20261113-0104/`):

- `test_results.json`: every number above, plus the validation comparison
- `test_scores.csv`: each test image's row_id, label, generator or source, type, style, file size, logit and calibrated estimate
- `reliability_test.png`: estimate vs. how often it's really AI, raw and calibrated
- `test_errors/`: `false_alarms_docci_top.png`, `false_alarms_imagenet_top.png`, `missed_ai_top.png`

The prepared test images and their CLIP features are in `RealEye/data/image/` (`test-*.parquet`, `features/test.npy`); the test set's counts and skips are in `manifest.json` under `test`. Data and models are not stored in git.
