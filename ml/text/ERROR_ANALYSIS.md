# Text Classifier — Error Analysis and Fixes

**Branch:** `ml-training` · **Started:** Oct 29, 2026 · **Follows:** [INITIAL_TRAINING_RESULTS.md](INITIAL_TRAINING_RESULTS.md)

## Summary

The first model (Oct 26, DistilRoBERTa, 10k human + 10k AI texts) flagged **41% of human texts as AI**. We needed a 0.999 cutoff to get that under 10%.

We analyzed its mistakes and found two causes:

1. **The training data was lopsided by writing type.** Human training texts were mostly Yelp reviews and Q&A. AI texts were spread evenly across all ten writing types. So the model partly learned "news summary / science abstract / story = AI."
2. **The model is overconfident.** When it's wrong, it's usually very sure: half of its wrong answers came with over 99% confidence.

We changed the data prep and training scripts to address both, then retrained (Oct 29, model `distilroberta-base-20261030-0247`). At the 0.5 cutoff, **false positives dropped from 41% to 26%** and **F1 rose from 0.818 to 0.874**. The model is no longer wrong with 99%+ confidence. With a new cutoff (about 0.94–0.947), validation reaches a 5–10% false-positive rate with F1 above 0.90, meeting both targets on validation. Next: calibrate the scores, choose the cutoff properly, and confirm on the test set. See [Results after retraining](#4-results-after-retraining) and [Next steps](#next-steps).

## Status

| Step | Status |
|---|---|
| Analyze the Oct 26 model's mistakes (`analyze_errors.py`) | Done |
| Check the training data mix by writing type | Done — found the imbalance |
| Balance training data by writing type (`prepare_data.py`) | Done, tested on dummy data |
| Label smoothing + best-epoch-by-loss (`train_text.py`) | Done, tested on dummy data |
| Retrain in Colab with the changes | Done — `distilroberta-base-20261030-0247` |
| Re-run `analyze_errors.py` on the new model and compare | Done — see section 4 |
| Write `calibrate.py` | Done, tested on synthetic scores and a dummy model |
| Run `calibrate.py` on the new model | **To do** |
| Final check on the held-out test set | **To do** — run only once, at the end |

## 1. What we found

All numbers are from the Oct 26 model on the 5,000-text validation set, produced by [analyze_errors.py](analyze_errors.py).

### Mistakes by writing type

MAGE texts come from ten sources (the `src` column), e.g. `yelp` = restaurant reviews, `xsum` = BBC news summaries.

| Writing type | What it is | Humans flagged as AI @0.5 | @0.999 | Share of all false positives |
|---|---|---|---|---|
| xsum | BBC news summaries | **80%** | **22%** | **24%** |
| tldr | Short summaries | 62% | 19% | 16% |
| hswag | Short everyday scenarios | 58% | 1% | 5% |
| sci_gen | Science paper abstracts | 46% | 11% | 12% |
| roct | Short stories | 37% | 2% | 9% |
| wp | Reddit writing prompts | 36% | 9% | 10% |
| squad | Wikipedia passages | 30% | 4% | 8% |
| cmv | Reddit arguments | 26% | 7% | 6% |
| eli5 | Reddit "explain like I'm 5" | 22% | 7% | 7% |
| yelp | Restaurant reviews | 19% | 3% | 4% |

- **Every writing type has the problem.** It isn't just one bad source.
- **Short, formal, factual writing is worst** (xsum, tldr, sci_gen). This is the kind of writing that most resembles AI output.
- **Raising the cutoff fixes some types but not others.** hswag drops from 58% to 1%, so those were low-confidence mistakes. xsum and tldr stay near 20%, so those were *confident* mistakes, and a cutoff can't fix them.
- **The model almost never misses AI text** (0–5% missed per type). It leans strongly toward saying "AI."

### How confident it is when wrong

Out of 5,000 texts, it got 1,090 wrong at the 0.5 cutoff. Of those wrong answers:

| Confidence in the wrong answer | Share of wrong answers |
|---|---|
| Over 90% | 75% |
| Over 99% | 51% |
| Over 99.9% | 21% |

A well-calibrated model would almost never be wrong at 99% confidence. This is why the cutoff ended up at 0.999, and why the "Model estimate: X%" in the extension popup would be misleading as-is.

### Why: the training data mix

`prepare_data.py` balanced human vs. AI **overall**, but not **within each writing type**. Share of each class coming from each writing type, in `train.jsonl`:

| Writing type | % of AI texts | % of human texts |
|---|---|---|
| yelp | 8.9% | **33.4%** |
| squad | 9.0% | 19.3% |
| eli5 | 11.4% | 17.6% |
| wp | 10.7% | 6.9% |
| xsum | 12.8% | **5.4%** |
| sci_gen | 8.6% | 5.3% |
| cmv | 9.2% | 5.0% |
| tldr | 9.4% | 3.3% |
| roct | 10.6% | 3.0% |
| hswag | 9.4% | **0.8%** |

The pattern matches the mistakes. Writing types with many human examples (yelp, eli5) have the fewest false positives. Types where AI examples far outnumber human ones (xsum, tldr, hswag) have the most. The model learned the writing *type* as a shortcut instead of learning what makes text AI-written.

(This was measured on the current 40k-per-class `train.jsonl`. The Oct 26 run used a 10k-per-class sample drawn the same way, so its mix would be about the same.)

### Validation loss per epoch

Not available for the Oct 26 model. The record was in the `checkpoints` folder, which the Colab instructions delete before saving to Drive. New training runs now save it in `metrics.json` (see below).

## 2. What we changed

| File | Change | Why |
|---|---|---|
| [prepare_data.py](prepare_data.py) | Training data now has equal human and AI texts **within each writing type**, up to 4,000 per class per type. Old behavior: `--balance label`. | Removes the "writing type = AI" shortcut. Types with few human texts (e.g. hswag) use all they have, so the train total comes out under 40k per class. |
| [train_text.py](train_text.py) | **Label smoothing** (`--label-smoothing`, default 0.1). | Trains toward 95% / 5% instead of 100% / 0%, so the model stops pushing scores to 0.999+. |
| [train_text.py](train_text.py) | Keeps the epoch with the **lowest validation loss** instead of the highest F1. | F1 is already above target. Overconfidence is the problem, and validation loss is what measures it. |
| [train_text.py](train_text.py) | Saves loss and metrics for every epoch in `metrics.json` (`log_history`). | So we can see whether a second epoch helps or makes the model more overconfident. |
| [analyze_errors.py](analyze_errors.py) | New script (see below). | Repeatable error analysis for any saved model. |
| [calibrate.py](calibrate.py) | New script: temperature scaling, cutoff choice, ECE, reliability chart (see section 5). | Makes scores honest and picks the cutoff without grading on the same texts. |
| [mage.py](mage.py) | Added `domain()`, which gets the writing type from `src`. | Shared by `prepare_data.py` and `analyze_errors.py`. |
| [COLAB.md](COLAB.md) | Updated the expected `prepare_data.py` output. | The train size now depends on the writing-type balance. |

### What label smoothing does

Without it, each training text has a hard target: AI = 100%, human = 0%. The model keeps getting pushed toward 100% even after it's right, so it learns to be extremely sure about everything, including texts it gets wrong.

With label smoothing at 0.1, the targets become 95% and 5%. Once the model reaches 95%, pushing further makes its score *worse*, so it stops piling scores up at the extremes. It doesn't change *which* texts the model thinks are AI. That's what the data balancing is for.

## 3. How to run the analysis

`analyze_errors.py` scores the validation set with a saved model and prints:

1. False-positive rates by writing type, at each cutoff
2. The 10 human texts the model thought were most AI-like
3. How confident the model is when it's wrong
4. Validation loss per epoch (models trained after Oct 29 only)

It also saves every score to `validation_scores.csv` in the model folder. Open that in Google Sheets and filter to `label = 0` to read the human texts it got wrong.

In Colab, from the `ml` folder, after `prepare_data.py`:

```
!python text/analyze_errors.py --model-dir /content/drive/MyDrive/RealEye/models/<model folder>
```

Takes about 2–3 minutes on a T4 GPU. Use `--thresholds 0.5 0.9 0.999` to compare other cutoffs.

### Colab setup notes

- Install with `!pip install -q -r requirements.txt`. It can take 10–15 minutes and looks frozen, because `-q` hides progress. Let it finish.
- The red "dependency conflicts" warnings afterward are normal. Only `torchvision` matters: it breaks `transformers`. Fix it with `!pip uninstall -y torchvision`, then **Runtime → Restart session**, then `%cd /content/realeye/ml`.
- If the `RealEye` Drive folder was shared with you, add a shortcut to it in **My Drive** first. Otherwise Colab can't find it by path.

## 4. Results after retraining

**Model:** `My Drive/RealEye/models/distilroberta-base-20261030-0247/` (trained Oct 29 on Colab T4, about 32 minutes)

**Settings:** DistilRoBERTa, data balanced by writing type (about 33,700 texts per class), label smoothing 0.1, 2 epochs, best epoch chosen by validation loss. Everything else is the same as Oct 26.

### Overall, at cutoff 0.5 (validation set)

| Metric | Oct 26 model | New model |
|---|---|---|
| False-positive rate | 41.5% | **26.4%** |
| F1 | 0.818 | **0.874** |
| Precision | 70.2% | 78.8% |
| Recall (AI caught) | 97.9% | 98.2% |
| Accuracy | 78.2% | 85.9% |
| Wrong answers (of 5,000) | 1,090 | 706 |

### Humans flagged as AI at 0.5, by writing type

| Writing type | Oct 26 | New |
|---|---|---|
| xsum | 80% | **48%** |
| tldr | 62% | 32% |
| hswag | 58% | 19% |
| sci_gen | 46% | 32% |
| roct | 37% | 9% |
| wp | 36% | 23% |
| squad | 30% | 28% |
| cmv | 26% | 16% |
| eli5 | 22% | 22% |
| yelp | 19% | 26% |

- **Balancing by writing type worked.** The types that were short on human examples improved the most, and the spread between types narrowed from 19–80% to 9–48%.
- **yelp got slightly worse.** That's expected: it used to be a third of the human training texts and now gets an equal share.
- **xsum (news summaries) is still the weakest type**, with 48% flagged and 23% of all false positives.

### Confidence when wrong

| Confidence in the wrong answer | Oct 26 | New |
|---|---|---|
| Over 90% | 75% | 72% |
| Over 99% | 51% | **0%** |
| Over 99.9% | 21% | **0%** |

- Label smoothing stopped the extreme scores. **The new model's scores top out around 0.957**, the ceiling label smoothing creates.
- **The old 0.999 cutoff no longer applies.** No text reaches it, so the model would call everything human. The cutoff must be chosen again for this model.
- Mistakes still cluster near the top of the new range (72% over 90%). The scale is compressed, but the scores aren't honest probabilities yet, so calibration is still needed.

### Cutoff preview (before calibration)

Cutoffs chosen so a set share of human validation texts get flagged (from `validation_scores.csv`):

| Target false-positive rate | Cutoff | F1 | Recall (AI caught) |
|---|---|---|---|
| 20% | 0.8932 | 0.893 | 96.8% |
| 10% | 0.9402 | **0.910** | 91.8% |
| 5% | 0.9469 | **0.905** | 86.8% |

For comparison, the Oct 26 model at its 0.999 cutoff had F1 0.826 with a 9.3% false-positive rate.

- **Both targets are met on validation with room to spare.** F1 stays above 0.90 even at 5% false positives.
- **Aim closer to 5% than 10%.** It costs almost no F1 and leaves room in case the test set comes out worse than validation.
- **These are optimistic.** Each cutoff is chosen and scored on the same texts. `calibrate.py` and the test set give the real numbers.
- **The cutoffs sit in a narrow band (0.89–0.95),** so small changes swing the false-positive rate a lot. Calibration should spread the scores out.

### Validation loss per epoch

| Epoch | Validation loss | F1 @0.5 |
|---|---|---|
| 1 | 0.622 | 0.834 |
| 2 | **0.508** | 0.874 |

Loss went down, so epoch 2 helped and was kept. It was still improving, so 3 epochs may be worth trying. These numbers can't be compared directly with the Oct 26 loss (1.01), because label smoothing changes how loss is calculated.

## 5. Calibration (`calibrate.py`)

**Why:** the new model's scores are squeezed into a narrow band (most between 0.85 and 0.957), so:
- the cutoff is fragile: 0.940 vs 0.947 swings false positives from 10% to 5%
- the "Model estimate: X%" in the extension doesn't mean what it says

**What temperature scaling does:** the model's raw output for each text (the *margin*: positive leans AI, negative leans human) is divided by one number, **T**, before it becomes a percentage. T > 1 pulls scores toward 50% (fixes a too-sure model). T < 1 spreads them out (fixes a too-timid one). T is chosen so the scores match reality as closely as possible.

Dividing by T never changes which text scores higher than which, so **it can't change F1 or the false-positive rate**. It only changes the numbers, so the cutoff lands somewhere stable and the percentage users see is honest.

**How the script avoids grading its own work:** the 5,000 validation texts are split into two halves with the same mix of human/AI and writing types.

| Step | Done on | Checked on |
|---|---|---|
| Fit T | Half A | Calibration error (ECE) measured on half B |
| Choose the cutoff (lowest one with false positives ≤ `--target-fpr`, default 5%) | Half B | F1 and false-positive rate measured on half A |

**What it prints:**
1. T, plus calibration error (ECE) before and after. ECE is the average gap between the score and how often it's right: 0 is perfect, 0.15 means off by 15 points.
2. A table of score groups (0–10%, 10–20%, …) showing what share of each group is actually AI, before and after.
3. The chosen cutoff, and the raw score it corresponds to.
4. F1, false-positive rate, recall and precision on the other half, plus false positives by writing type.
5. The same check for 10%, 7%, 5% and 3% false-positive targets, for comparison.

**What it saves** (in the model folder):
- `calibration.json`: T, the cutoff and the metrics. The server uses T and the cutoff: `ai_score = 1 / (1 + exp(-margin / T))`, labeled AI if `ai_score >= cutoff`.
- `reliability.png`: predicted score vs. share actually AI, before and after. A perfectly calibrated model falls on the diagonal. Good for the status report.

To try a different false-positive target: `--target-fpr 0.07`.

## Next steps

1. **Calibrate** — run `calibrate.py` on the new model (see section 5):
   ```
   !python text/calibrate.py --model-dir /content/drive/MyDrive/RealEye/models/distilroberta-base-20261030-0247
   ```
2. **Final test-set check**, run once: F1, false-positive rate and ECE at the chosen cutoff. The targets are F1 ≥ 0.75 and false-positive rate ≤ 0.10 (SRS 5.4.8, 5.4.20).
3. If false positives are still too high, try training for 3 epochs (validation loss was still dropping), or `FacebookAI/roberta-base` (`--model FacebookAI/roberta-base`), as planned in [DATASETS.md](DATASETS.md).
