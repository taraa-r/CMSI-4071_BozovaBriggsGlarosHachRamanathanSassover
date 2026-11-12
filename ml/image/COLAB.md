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

### B2. Label styles and cap AI illustrations

```
!python image/label_styles.py --data-dir /content/drive/MyDrive/RealEye/data/image
```

Look at the "before" and "after" style tables for each split. After the cap, the **illustration** share for AI should be close to the real share. The `Kept N real + N AI` lines show how many images remain (the small test run lost about 6% of train).

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

## Troubleshooting

| Problem | Fix |
|---|---|
| Session disconnected during A3 | Redo A2, then run A3 again: it resumes. Keep the tab open and in front. |
| Many `Read timed out` / `Retrying in ...` lines or `429` errors | Normal now and then, as long as progress lines keep appearing. If none appear for 5+ minutes, stop the cell and rerun A3 as `!HF_HUB_DOWNLOAD_TIMEOUT=60 python image/prepare_data.py --out-dir /content/drive/MyDrive/RealEye/data/image --workers 4` (it resumes). |
| `Skipped a row group that failed to download` | Fine now and then; the run reads more chunks to make up for it. |
| `styles.parquet not found` | Run B2 before B3. |
| `CUDA out of memory` in B2 or B3 | Add `--batch-size 32` (B3); for B2, edit `BATCH = 64` to 32 in `label_styles.py`. |
| Scripts say `cpu` instead of `cuda` in Part B | Runtime → Change runtime type → T4 GPU, then redo A2. |
| `operator torchvision::nms does not exist` | Run `!pip uninstall -y torchvision`, then **Runtime → Restart session**, then `%cd /content/realeye/ml`. |
| "Cannot connect to GPU backend" | Free GPU time is used up for now. Try later or with another Google account. Part A needs no GPU. |
