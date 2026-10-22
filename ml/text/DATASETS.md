# Text Model — Base Models and Datasets

Source and license record for the text classifier (SRS 5.3.4.3, 5.3.4.7).
Owners: Noa, Tara. Decided: October 22, 2026.

## Label convention

**0 = human, 1 = AI-generated** — for every dataset, after loading. Some sources use the opposite (see MAGE).

## Base models

| Role | Model | License | Size | Why |
|---|---|---|---|---|
| **Primary** | [`distilbert/distilroberta-base`](https://huggingface.co/distilbert/distilroberta-base) | Apache-2.0 | 82M parameters | Distilled RoBERTa: ~2× faster than `roberta-base`, keeps capitalization, pretrained on web text. Fast enough for quick iteration. |
| Comparison | [`FacebookAI/roberta-base`](https://huggingface.co/FacebookAI/roberta-base) | MIT | 125M parameters | Standard baseline in AI-text detection research. Try it if `distilroberta-base` lands near the F1 target; keep whichever scores better on **validation** data. |

Considered and not chosen: `distilbert-base-uncased` — lowercases all text (loses style signals) and was pretrained on books + Wikipedia rather than web text.

## Datasets

| Role | Dataset | License | Size | Notes |
|---|---|---|---|---|
| **Train / validation / test** | [MAGE](https://huggingface.co/datasets/yaful/MAGE) (`yaful/MAGE`) | Apache-2.0 | 319k train · 56.8k validation · 60.7k test | Many AI generators and topics. Already split — use its splits as-is. **Its `label` is 1 = human: flip it on load.** |
| Extra generalization test (optional) | [RAID](https://huggingface.co/datasets/liamdugan/raid) (`liamdugan/raid`) | MIT | ~5.4M train rows | 12 generators, includes adversarial attacks. Use a sample only for a final out-of-distribution check, never for training or threshold selection. |

Considered and not chosen: [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) (CC-BY-SA-4.0) — its human answers have tokenization artifacts (e.g. `word , word`) that a model can learn as a shortcut instead of real differences.

## Preprocessing (to match the extension and SRS 5.5.3)

- Normalize whitespace, then keep only texts of **200–5000 characters**.
- English only.
- Remove exact duplicates, and make sure no text appears in more than one split.

## Known limitations

- MAGE and RAID mostly contain text from 2023-era models. Newer generators may be harder to detect — report this in the SDD and final evaluation.
- Results describe this test data, not every website or AI model (SRS 5.5.3).
