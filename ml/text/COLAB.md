# Training the Text Model on Google Colab

Colab gives a free GPU in the browser. The full training run takes roughly 20–40 minutes there vs. 8+ hours on a laptop CPU.

**Important:** when a Colab session ends, everything on its machine is deleted. Step 6 copies the trained model to your Google Drive — don't skip it.

## 1. Open Colab with a GPU

1. Go to https://colab.research.google.com and sign in with your Google account.
2. **File → New notebook in Drive**. Rename it (top left) to `RealEye text training`.
3. **Runtime → Change runtime type → T4 GPU → Save**.

Each step below goes in its own code cell: click **+ Code**, paste, then press **Shift+Enter** to run it.

## 2. Check the GPU

```
!nvidia-smi
```

You should see a table naming a **Tesla T4**. If it says the command was not found, redo step 1.3.

## 3. Connect Google Drive

```
from google.colab import drive
drive.mount('/content/drive')
```

A pop-up asks for permission — allow it. This is where the trained model will be saved.

## 4. Get the code and install packages

```
!git clone https://github.com/taraa-r/CMSI-4071_BozovaBriggsGlarosHachRamanathanSassover.git realeye
%cd realeye/ml
!pip install -q -r requirements.txt
```

This takes a few minutes. If Colab shows a **"Restart session"** button afterward, click it, then run this before continuing:

```
%cd /content/realeye/ml
```

## 5. Prepare the data and train

```
!python text/prepare_data.py
```

Should end with `train 20,000 rows`, `validation 5,000 rows`, `test 51,002 rows` — the same numbers as on a laptop (seed 42).

```
!python text/train_text.py
```

Progress bars show the training. When it finishes, it prints a line like:

```
Validation (cutoff 0.5): F1 0.xxx | accuracy 0.xxx | false-positive rate 0.xxx
```

Write those three numbers down and share them with the team. The goal for the final model is **F1 ≥ 0.75** and **false-positive rate ≤ 0.10** (SRS 5.4.8, 5.4.20).

## 6. Save the model to Google Drive

```
!mkdir -p /content/drive/MyDrive/RealEye/models
!rm -rf models/text/*/checkpoints
!cp -r models/text/* /content/drive/MyDrive/RealEye/models/
!cp data/text/manifest.json /content/drive/MyDrive/RealEye/models/
!ls /content/drive/MyDrive/RealEye/models
```

You should see a folder like `distilroberta-base-20261023-1415`. It contains the model (~330 MB) and `metrics.json`. Share the `RealEye` Drive folder with your teammate.

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Run `!python text/train_text.py --batch-size 8` |
| "Cannot connect to GPU backend" | Free GPU time is used up for now. Try again later or with another Google account. |
| Session disconnected mid-training | Start again from step 3. Keep the tab open and in front while it trains. |
| Want to compare with RoBERTa | `!python text/train_text.py --model FacebookAI/roberta-base` (about 2× slower) |
