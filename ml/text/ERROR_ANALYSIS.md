# Text Classifier — Error Analysis and Fixes

**Branch:** `ml-training` · **Started:** Oct 29, 2026 · **Follows:** [INITIAL_TRAINING_RESULTS.md](INITIAL_TRAINING_RESULTS.md)

## Summary

The first model (Oct 26, DistilRoBERTa, 10k human + 10k AI texts) flagged **41% of human texts as AI**. We needed a 0.999 cutoff to get that under 10%.

We analyzed its mistakes and found two causes:

1. **The training data was lopsided by writing type.** Human training texts were mostly Yelp reviews and Q&A. AI texts were spread evenly across all ten writing types. So the model partly learned "news summary / science abstract / story = AI."
2. **The model is overconfident.** When it's wrong, it's usually very sure: half of its wrong answers came with over 99% confidence.

We changed the data prep and training scripts to address both. **The retrained model has not been run yet.** See [Next steps](#next-steps).

## Status

| Step | Status |
|---|---|
| Analyze the Oct 26 model's mistakes (`analyze_errors.py`) | Done |
| Check the training data mix by writing type | Done — found the imbalance |
| Balance training data by writing type (`prepare_data.py`) | Done, tested on dummy data |
| Label smoothing + best-epoch-by-loss (`train_text.py`) | Done, tested on dummy data |
| Retrain in Colab with the changes | **To do** |
| Re-run `analyze_errors.py` on the new model and compare | **To do** |
| Calibrate scores and pick the final cutoff (`calibrate.py`) | **To do** — not written yet |
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

## Next steps

1. **Retrain** in Colab on `ml-training`: `prepare_data.py`, then `train_text.py`. Expect roughly 30–60 minutes. Save the model to Drive (COLAB.md step 6).
2. **Re-run `analyze_errors.py`** on the new model and compare it with the tables above. What to look for:
   - The xsum and tldr false-positive rates should drop.
   - Fewer wrong answers should come with over 99% confidence.
   - Validation loss should not rise from epoch 1 to epoch 2.
3. **Calibrate** (`calibrate.py`, not written yet). Fit temperature scaling so a score of 80% means about 80%. Pick the cutoff on calibrated scores. Report expected calibration error (ECE) and a reliability chart. Use separate halves of the validation set for fitting and for picking the cutoff, so the results aren't optimistic.
4. **Final test-set check**, run once: F1, false-positive rate and ECE at the chosen cutoff. The targets are F1 ≥ 0.75 and false-positive rate ≤ 0.10 (SRS 5.4.8, 5.4.20).
5. If false positives are still too high, try `FacebookAI/roberta-base` (`--model FacebookAI/roberta-base`), as planned in [DATASETS.md](DATASETS.md).
