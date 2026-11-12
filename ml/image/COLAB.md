# Image Model on Google Colab: Data Prep and CLIP Training

This builds the full image dataset from OpenFake and trains the CLIP classifier. It runs in two parts:

| Part | What | Runtime | Time (rough) |
|---|---|---|---|
| A | Prepare the data, check shortcuts | **CPU** (no GPU needed) | 2–4 hours, mostly downloading ~80 GB |
| B | Label styles, CLIP features, train the classifier | **T4 GPU** | 20–40 minutes |

Everything is saved to Google Drive as it goes, so a disconnect doesn't lose finished work. Part A runs on a CPU runtime because it only downloads and resizes images; this saves your limited free GPU time for Part B.

**Before you start:** the image scripts must be on GitHub, because Colab gets the code with `git clone`. Make sure the latest `ml/image/` files are committed and pushed to the branch you clone.

Each step goes in its own code cell: click **+ Code**, paste, then press **Shift+Enter** to run it.

## Part A: prepare the data (CPU runtime)

### A1. Open Colab

1. Go to https://colab.research.google.com and sign in.
2. **File → New notebook in Drive**. Rename it to `RealEye image data`.
3. Leave the runtime as **CPU** (Runtime → Change runtime type → CPU).

### A2. Connect Google Drive and get the code

```
from google.colab import drive
drive.mount('/content/drive')
```

Allow the pop-up. Then:

```
!git clone https://github.com/taraa-r/CMSI-4071_BozovaBriggsGlarosHachRamanathanSassover.git realeye
%cd realeye/ml
!pip install -q -r requirements.txt
```

Pip prints warnings about dependency conflicts (diffusers, gradio, gcsfs, torchvision). They are harmless except torchvision, which would break the CLIP scripts in Part B. Remove it:

```
!pip uninstall -y torchvision
```

Then click **Runtime → Restart session**, and run:

```
%cd /content/realeye/ml
```

### A3. Prepare the data

```
!python image/prepare_data.py --out-dir /content/drive/MyDrive/RealEye/data/image
```

It prints a progress line about every 16 chunks:

```
  160 row groups (13.9 GB): real 5,212 / AI 5,190 of 26,000 each
```

It stops when both counts reach 26,000 (30% extra, so balancing can trim the most common generators), then does validation the same way (3,250 each).

**If the session disconnects** (common after an hour or two, or if the tab sits in the background): redo A2, then run the same A3 command again. It skips finished splits and continues an unfinished one from its last saved batch. You'll see `Resuming an unfinished run: N batches...`.

At the end, check these lines:

```
  Kept 20,000 real + 20,000 AI images from NN generators. Skipped: ...
  Kept 2,500 real + 2,500 AI images from NN generators. Skipped: ...
```

If a split kept fewer than requested, note the numbers and tell the team.

### A4. Recheck the shortcuts

```
!python image/check_shortcuts.py --data-dir /content/drive/MyDrive/RealEye/data/image
```

This is the full-size recheck of the file-size gap. For train and validation, look at:

- The table: **JPEG 100%** for both groups; **EXIF 0%**; **Square 100%**; similar median width and JPEG quality.
- The AUC lines: each should say `ok` (between 0.45 and 0.55).
- **File size AUC on validation:** the small test run gave 0.58. Write down the new value. If it says `SHORTCUT LEFT` again, that's not a blocker: the blur and JPEG augmentation in Part B is meant to handle it, and train_clip.py reports whether scores follow file size.

Share the printout with the team, and add the numbers to the shortcut table in `DATASETS.md`.

## Part B: CLIP training (GPU runtime)

### B1. Switch to a GPU

**Runtime → Change runtime type → T4 GPU → Save**, then check it:

```
!nvidia-smi
```

You should see a table naming a **Tesla T4**. Switching runtimes resets the machine, so redo **A2** (mount Drive, clone, install). The data from Part A is safe on Drive.

### B2. Label styles and match them between real and AI

```
!python image/label_styles.py --data-dir /content/drive/MyDrive/RealEye/data/image
```

For train, look at the "before" and "after" style tables: after matching, every style group has the same count for real and AI. Validation is not matched and keeps every image (it says `Not matched: all images kept`). The `Kept N real + N AI` lines show how many images remain. (Until Nov 12 this step only capped AI illustrations; see `ERROR_ANALYSIS.md` for why it now matches every style in train.)

### B3. Turn the images into CLIP features

```
!python image/extract_features.py --data-dir /content/drive/MyDrive/RealEye/data/image
```

This runs CLIP over every train image three times (once clean, twice with random blur and/or extra JPEG compression) and every validation image once. The features go to `.../data/image/features/` (about 150 MB).

### B4. Train the classifier

```
!python image/train_clip.py --data-dir /content/drive/MyDrive/RealEye/data/image --out-dir /content/drive/MyDrive/RealEye/models/image/clip-linear-$(date +%Y%m%d-%H%M)
```

This takes a few minutes. It tries four weight-decay settings and keeps the one with the lowest validation loss. If it prints `still improving at the last round`, rerun with `--epochs 60` added.

Then it prints a line like:

```
Validation (first-look cutoff 0.5): F1 0.xxx | accuracy 0.xxx | false-positive rate 0.xxx | AUC 0.xxx
```

Write these down and share them with the team. The project targets are **F1 ≥ 0.75** and **false-positive rate ≤ 0.10** (SRS 5.4.8, 5.4.20). The 0.5 cutoff is only a first look: the real cutoff is chosen in the calibration step, as for the text model.

It also prints two checks:

- **False-positive rate by real source** (LAION vs. Pexels): a big gap means the model treats one kind of real photo as more "AI-like".
- **Score vs. file size (Spearman)**: near 0 means scores don't just follow image detail. Above about 0.3 for real images would mean sharp real photos tend to get flagged.

The model folder on Drive holds `head.pt` (the trained layer, a few KB), `config.json`, `metrics.json` (the full breakdowns by generator and style) and `validation_scores.csv` (for calibration). Share the `RealEye` Drive folder with your teammate.

### B5. Look at the mistakes (CPU is fine)

```
!python image/analyze_errors.py --model-dir /content/drive/MyDrive/RealEye/models/image/<model folder> --data-dir /content/drive/MyDrive/RealEye/data/image
```

Replace `<model folder>` with the folder B4 printed (e.g. `clip-linear-20261112-0823`). This takes about a minute and doesn't rerun CLIP. The cutoff is placed where 10% of real images get flagged (change it with `--target-fpr 0.08`).

It prints three things:

1. **Real images wrongly flagged**, by source (LAION, Pexels) and by CLIP style. The `share of all` column shows where most false alarms come from.
2. **AI images caught**, by style and by generator, worst first.
3. **How sure the model is when it's wrong**, and whether false alarms are smaller files or look less like photos to CLIP.

It also saves contact sheets to `<model folder>/errors/`. Each image is labeled with its score, style and file size. Open them in the Drive file browser:

- `false_alarms_laion_top.png` and `false_alarms_laion_random.png`: the real LAION images the model calls AI. Compare them with `correct_laion_random.png`.
- The same three sheets for Pexels.
- `missed_ai_top.png`: the AI images the model is most sure are real.

`errors.csv` lists every mistake. Share the printout and the sheets with the team before changing the data or the model.

## Part C: retrain with style matching (Nov 12)

The first model (`clip-linear-20261112-0823`) was trained with only AI illustrations capped. `ERROR_ANALYSIS.md` explains why every style is now matched. To redo it on the existing data (Part A doesn't need to run again):

### C1. Get the new code (T4 GPU runtime)

Switch to a **T4 GPU** (Runtime → Change runtime type), because C3 runs CLIP over every kept image. Then mount Drive and get the code as in A2. If the `realeye` folder is still there, update it instead:

```
%cd /content/realeye
!git pull
%cd /content/realeye/ml
```

On a fresh runtime, also do the pip install and torchvision removal from A2.

### C2. Match styles

```
!python image/label_styles.py --data-dir /content/drive/MyDrive/RealEye/data/image
```

It says `Reusing the CLIP style scores` and takes seconds. Write down the `Kept N real + N AI (dropped ...)` line for train. Validation should say `dropped 0`.

### C3. Rebuild the CLIP features for the kept images

```
!python image/extract_features.py --data-dir /content/drive/MyDrive/RealEye/data/image
```

This replaces the old features, and takes about as long as B3 did, a bit less since fewer images are kept.

### C4. Train and compare with the first model

```
!python image/train_clip.py --data-dir /content/drive/MyDrive/RealEye/data/image --out-dir /content/drive/MyDrive/RealEye/models/image/clip-linear-$(date +%Y%m%d-%H%M) --compare-to /content/drive/MyDrive/RealEye/models/image/clip-linear-20261112-0823
```

Besides the usual output, it prints an `earlier` and a `new` line. Both models are scored on the **same** new validation images, at 0.5 and at the cutoff that flags 10% of real images. Compare:
- recall and F1 at that cutoff;
- LAION vs. Pexels false-positive rate;
- recall by style, especially `photo`.

### C5. Look at the mistakes again

Run B5 with the new model folder that C4 printed. Share all the printouts and the LAION contact sheets, and add the results to `ERROR_ANALYSIS.md` ("Results after style matching").

## Troubleshooting

| Problem | Fix |
|---|---|
| Session disconnected during A3 | Redo A2, then run A3 again: it resumes. Keep the tab open and in front. |
| Many `Read timed out` / `Retrying in ...` lines or `429` errors | Normal now and then, as long as progress lines keep appearing. If none appear for 5+ minutes, stop the cell and rerun A3 as `!HF_HUB_DOWNLOAD_TIMEOUT=60 python image/prepare_data.py --out-dir /content/drive/MyDrive/RealEye/data/image --workers 4` (it resumes). |
| `Skipped a row group that failed to download` | Fine now and then; the run reads more chunks to make up for it. |
| `styles.parquet not found` | Run B2 before B3. |
| `The train features don't match the images kept in styles.parquet` | `label_styles.py` was rerun after the features were built. Run `extract_features.py` (B3/C3) again, then train. |
| `CUDA out of memory` in B2 or B3 | Add `--batch-size 32` (B3); for B2, edit `BATCH = 64` to 32 in `label_styles.py`. |
| Scripts say `cpu` instead of `cuda` in Part B | Runtime → Change runtime type → T4 GPU, then redo A2. |
| `operator torchvision::nms does not exist` | Run `!pip uninstall -y torchvision`, then **Runtime → Restart session**, then `%cd /content/realeye/ml`. |
| "Cannot connect to GPU backend" | Free GPU time is used up for now. Try later or with another Google account. Part A needs no GPU. |
