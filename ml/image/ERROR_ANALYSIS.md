# Image Classifier — Error Analysis

This looks at where the first full CLIP model (`clip-linear-20261112-0823`, see [INITIAL_TRAINING_RESULTS.md](INITIAL_TRAINING_RESULTS.md)) goes wrong on the validation set, and explains the fix chosen: **style matching**. All numbers are from validation (2,227 real + 2,227 AI images). The test set has not been used.

## Summary

The model has partly learned **"doesn't look like a normal photo = AI."** That one pattern explains both of its weak spots:

- **Real images that aren't ordinary snapshots get flagged:** human-made digital art, CGI, posters, and glossy studio or promo photos.
- **AI images that look like ordinary photos slip through:** photorealistic AI is the hardest to catch, and it's the kind that matters most for misinformation.

The cause is in the data. After the Nov 11 illustration cap, **graphics and 3D renders were still about twice as common among AI images as among real ones**, so the model could use style as a shortcut. The fix is to match every style between real and AI, then retrain.

## How the analysis was done

1. **Cutoff tradeoff** (quick check in Colab): for target false-positive rates of 5%, 10% and 15%, find the cutoff that gives that rate and see what the model catches there.
2. **[analyze_errors.py](analyze_errors.py)**, at the cutoff that flags 10% of real images (0.684). It breaks down mistakes by real source (LAION, Pexels), by CLIP style (photo, illustration, render, graphic) and by AI generator. It also saves contact sheets of the mistakes, so they can be looked at directly.

Both use only the saved validation scores (`validation_scores.csv`). No retraining was involved.

## 1. Cutoff tradeoff

At the 0.5 first-look cutoff, the false-positive rate was 20.3% (target: ≤ 10%). The AUC of 0.906 means the model ranks well, so a stricter cutoff can lower false positives:

| Allowed false positives | Cutoff | Recall (AI caught) | F1 | LAION FPR | Pexels FPR | Meets both targets? |
| --- | --- | --- | --- | --- | --- | --- |
| 5% | 0.802 | 60.0% | 0.727 | 8.0% | 2.1% | No, F1 below 0.75 |
| **10%** | **0.684** | **73.1%** | **0.799** | **15.3%** | **4.8%** | **Yes** |
| 15% | 0.588 | 79.9% | 0.820 | 22.4% | 7.7% | No, FPR above 10% |

- **Both targets can be met at once** with a cutoff near 0.68 (F1 ≥ 0.75, false-positive rate ≤ 10%).
- **The cost of a stricter cutoff:** fewer real images are flagged, but more AI images get through. At 5%, the model misses 40% of AI images.
- **These numbers are slightly optimistic.** The cutoff was chosen and graded on the same images, so the 10% is exact by construction. The calibration step will choose the cutoff on one half of validation and measure it on the other half.
- **Aim slightly below 10%.** The text model aimed for 5% on validation and got 5.9% on test, so rates drift up on new data. The plan is to target about 8% in calibration and to fix the target before running the test set.
- **The LAION/Pexels gap doesn't change with the cutoff.** LAION is flagged about 3× as often at every cutoff. Calibration only moves one cutoff for all images, so it can't close this gap. The model itself has to change.

## 2. False alarms: real images flagged as AI

At the 10% cutoff, 223 of 2,227 real images are flagged.

| Source | Images | Flagged | % flagged | Share of all false alarms |
| --- | --- | --- | --- | --- |
| LAION | 1,103 | 169 | 15.3% | 76% |
| Pexels | 1,124 | 54 | 4.8% | 24% |

By source and CLIP style:

| Source | Style | Images | Flagged | % flagged | Share of all false alarms |
| --- | --- | --- | --- | --- | --- |
| LAION | graphic | 196 | 46 | 23.5% | 21% |
| LAION | illustration | 19 | 4 | 21.1% | 2% |
| LAION | photo | 828 | 106 | 12.8% | 48% |
| LAION | render | 60 | 13 | 21.7% | 6% |
| Pexels | graphic | 128 | 7 | 5.5% | 3% |
| Pexels | illustration | 12 | 1 | 8.3% | 0% |
| Pexels | photo | 860 | 36 | 4.2% | 16% |
| Pexels | render | 124 | 10 | 8.1% | 4% |

**Findings:**

- **Non-photo styles get flagged about twice as often** as photos within LAION (21–24% vs. 12.8%).
- **Style is only part of it.** LAION *photos* are still flagged 3× as often as Pexels photos (12.8% vs. 4.2%), and they make up the largest group of false alarms (48%).
- **Less photo-like means more likely flagged.** Asked how much an image looks like "a photo", CLIP gives false alarms lower scores than real images the model got right:

  | Source | Median CLIP "photo" score: false alarms | Median: correct |
  | --- | --- | --- |
  | LAION | 0.63 | 0.81 |
  | Pexels | 0.52 | 0.70 |

- **File size is not the cause.** LAION false alarms have a median of 25.7 KB vs. 24.6 KB for correct ones; Pexels 21.2 vs. 23.8 KB. The file-size gap from the shortcut check (INITIAL_TRAINING_RESULTS.md) is ruled out as an explanation.
- **Some mistakes are confident.** 17% of LAION and 19% of Pexels false alarms scored 0.9 or higher. Calibration can't fix those, because it doesn't change the order of the scores.

### What the contact sheets show

`errors/false_alarms_laion_top.png` (the 16 real LAION images the model is most sure are AI, scores 0.93–0.99) and `false_alarms_laion_random.png` (16 random LAION false alarms) show two kinds of images:

1. **Human-made digital or artistic images.** Vintage-style ad posters, a caricature painting, an anime film poster, a superhero movie still, a CGI motorcycle, a movie poster, digital fan-art paintings, magazine covers and collages. None of these are AI-generated, but they aren't camera photos either.
2. **Polished commercial photos.** Posed band and team promo shots, an airline crew ad, a wedding dress against a painted backdrop, bridesmaid catalogue photos, red-carpet and glamour portraits. Studio lighting, retouched skin and perfect composition are exactly the look modern AI generators produce.

`correct_laion_random.png` (real LAION images the model got right) is the contrast: candid news and sports photos, snapshots, product shots on white backgrounds, plain flyers and old black-and-white photos. These are ordinary, unpolished web images.

**Probably not label noise.** Almost none of the false alarms look AI-made. LAION was collected mostly before AI image generators became common, so these are genuinely hard real images, not mislabeled ones.

## 3. Misses: AI images that slip through

At the 10% cutoff, the model catches 1,629 of 2,227 AI images (73.1%).

| AI style | Images | Caught (cutoff 0.684) | Caught (cutoff 0.5) |
| --- | --- | --- | --- |
| Illustration | 81 | 93.8% | 97.5% |
| Render | 459 | 86.9% | 93.2% |
| Graphic | 430 | 80.0% | 89.8% |
| **Photo** | **1,257** | **64.4%** | **79.5%** |

**Photorealistic AI is the hardest to catch**, while stylized AI is easy. This is the mirror image of the false alarms: the model leans on "looks stylized" rather than on actual signs of AI.

**By generator** (cutoff 0.684, about 44 images each):

| Hardest | Caught | Easiest | Caught |
| --- | --- | --- | --- |
| qwen-image | 28.6% | studio-ghibli-wan2.1-t2v-14b-v1.0 | 100% (6 images) |
| stable-diffusion-v1-5 | 47.7% | cyberrealistic-pony-v14.0 | 97.7% |
| sdxl-realvis-v5 | 50.0% | novaanimexl_ilv140 | 96.6% |
| sd-2.1 | 51.6% | one-obsession-17-red-sdxl | 94.4% |
| flux-1.1-pro | 53.5% | animagine-xl-3.1 | 92.7% |
| stable-diffusion-v1-4 | 53.7% | cyberrealistic-pony-v15.0 | 92.5% |
| sd-3.5 | 54.8% | sd-xl-v1.0-vae-fix | 92.1% |
| sd-turbo | 56.8% | playground-v2.5-1024px-aesthetic | 90.9% |
| sdxl-turbo, midjourney-6 | 58.1% | kolors-v1.0 | 89.3% |
| flux-realism | 59.1% | animagine-xl-4.0 | 89.2% |

- **The hardest generators** are mostly realism-tuned models (realvis, flux-realism, flux-1.1-pro, midjourney-6), qwen-image, and the older Stable Diffusion 1.x/2.x models.
- **The easiest** are anime and art models (animagine, pony, novaanime, ghibli).
- **A possible reason the old SD models are hard:** they were trained on LAION, so their images may imitate LAION photos. This hasn't been checked.
- **Caveat:** with about 44 images per generator, differences of less than about 15 points between two generators could be chance. Read the table for patterns, not exact rankings.

The full per-generator list is in `metrics.json` (cutoff 0.5) and the `analyze_errors.py` printout (cutoff 0.684).

## 4. The cause: style imbalance left in the data

Style shares in the validation set used for this model (after the Nov 11 illustration cap):

| Style | Real | AI |
| --- | --- | --- |
| Photo | 1,688 (76%) | 1,257 (56%) |
| Graphic | 324 (15%) | 430 (19%) |
| Render | 184 (8%) | 459 (21%) |
| Illustration | 31 (1%) | 81 (4%) |
| **Not a photo, total** | **24%** | **44%** |

The illustration cap fixed illustrations only. **Renders and graphics were still much more common on the AI side.** The model learned what the data taught it: a non-photo look means probably AI. This is the same kind of shortcut as the file properties removed in data prep, one level up: the model can predict the label from *what kind of picture it is* instead of from signs of AI.

## 5. The fix: style matching (Nov 12)

**Decision:** match every style between real and AI, not just illustrations. In each style group, real and AI keep the **same number** of images (the larger side is trimmed at random), in both train and validation. A model trained this way can't use style as a shortcut, so it has to find signs of AI within each style.

**Style groups** (from CLIP's style scores, already saved in `styles.parquet`):

| Group | Rule |
| --- | --- |
| photo (clear) | best match "a photo", with CLIP's photo score ≥ 0.8 |
| photo (unclear) | best match "a photo", photo score < 0.8: polished, staged or edited-looking photos |
| illustration | best match "a drawing, painting, cartoon, anime or digital illustration" |
| render | best match "a 3D render or video game graphics" |
| graphic | best match "a poster, advertisement or graphic design with text" |

- **Why split photos in two:** false alarms had lower photo scores (0.63 vs. 0.81 for LAION). Real images skew toward clear photos and AI images toward unclear ones. Matching within each half stops the model from using "how photo-like" as a shortcut inside the photo style too.
- **Why imperfect style labels are fine:** CLIP sometimes labels a real studio photo as a "render". For matching that doesn't matter. The classifier only sees CLIP features, so balancing what CLIP perceives is exactly what removes the shortcut.

**Cost:** fewer images. On the small test run, matching dropped 12% of train and 27% of validation. On the full data, an estimated 20–25% of images will be dropped (about 15,000 per class left in train). The `label_styles.py` printout gives the real numbers. With a frozen encoder and one linear layer, 15,000 per class is still plenty.

**Code changes:**
- `label_styles.py` now matches all style groups instead of capping illustrations. It reuses the saved CLIP style scores, so it runs in seconds without a GPU (`--rescore` runs CLIP again).
- `extract_features.py` is unchanged but must be rerun, since it builds features only for kept images.
- `train_clip.py`:
  - It stops if the features don't match the kept images (prevents training on stale features).
  - It always reports results at the cutoff that flags 10% of real images.
  - New `--compare-to`: it scores the earlier model on the **same** validation images, for a fair before-and-after comparison. The two validation sets differ, so their headline numbers can't be compared directly.

**What it should change:**
- Fewer false alarms on LAION graphics and renders.
- Better recall on photo-style AI.
- Possibly slightly lower recall on stylized AI, which was inflated by the shortcut.

**What it won't fully fix:**
- Polished studio photos that CLIP still calls clear photos.
- Human-made digital art and CGI, which are genuinely hard.

Both stay known limitations for the write-up and the extension's wording. For example, a flagged image could be described as "may be AI-generated or heavily edited/digital art".

## Results after style matching

*To be filled in after the rerun (COLAB.md, Part C): the `label_styles.py` counts, the new validation results, the `--compare-to` table, and a new `analyze_errors.py` run.*

## Next steps

1. Rerun on Colab with style matching (COLAB.md, Part C) and fill in the section above.
2. Calibration (`image/calibrate.py`, modeled on `text/calibrate.py`), with a target false-positive rate of about 8% and the 10%/9%/8%/7%/5% comparison.
3. One-time test set evaluation: overall, by style, by real source (DOCCI, ImageNet) and on unseen generators.
4. Server integration.
