# Text Classifier — Initial Training Results

## Overview

The first full training run of RealEye's text classifier was completed successfully using **DistilRoBERTa** (`distilbert/distilroberta-base`) on Google Colab with GPU acceleration.

The model was fine-tuned on the MAGE dataset to classify text as either human-written (label 0) or AI-generated (label 1).

This training run establishes an initial baseline. The results below are from the **validation dataset**, not the final held-out test dataset.

## Training Configuration

| Setting                  | Value                     |
| ------------------------ | ------------------------- |
| Model                    | DistilRoBERTa             |
| Dataset                  | MAGE                      |
| Training examples        | 20,000 (10,000 per class) |
| Validation examples      | 5,000 (2,500 per class)   |
| Epochs                   | 2                         |
| Batch size               | 16                        |
| Learning rate            | 2e-5                      |
| Maximum sequence length  | 512 tokens                |
| Hardware                 | Google Colab GPU          |
| Training runtime         | Approximately 10 minutes  |
| Classification threshold | 0.5                       |

Training was executed using:

`python text/train_text.py`

## Initial Validation Results

| Metric              | Result |
| ------------------- | ------ |
| Accuracy            | 78.20% |
| Precision           | 70.24% |
| Recall              | 97.88% |
| F1 Score            | 0.818  |
| False-Positive Rate | 41.48% |
| Validation Loss     | 1.0144 |
| Training Loss       | 0.2523 |

### Explanation of Results

**Accuracy (78.20%)**

Accuracy represents the percentage of all validation examples correctly classified as either human-written or AI-generated. The model correctly classified approximately 78% of the validation dataset.

**Precision (70.24%)**

Precision measures how often the model is correct when it predicts that text is AI-generated. Approximately 70% of the passages classified as AI-generated were actually AI-generated.

**Recall (97.88%)**

Recall measures the percentage of actual AI-generated passages correctly detected. The model successfully identified almost 98% of AI-generated examples, indicating that it is effective at detecting AI-generated text.

**F1 Score (0.818)**

The F1 score combines precision and recall using their harmonic mean. It measures how effectively the model balances identifying AI-generated text with avoiding incorrect AI predictions. The score of 0.818 exceeds RealEye's target of 0.75.

**False-Positive Rate (41.48%)**

The false-positive rate measures how often human-written text is incorrectly classified as AI-generated. The model incorrectly flagged approximately 41% of human-written validation examples. This is significantly higher than RealEye's target of 10%.

**Training and Validation Loss**

The reported training loss was 0.2523, while the validation loss was 1.0144. These values measure the model's prediction errors during training and evaluation. The relatively high validation loss suggests that some incorrect predictions may be made with high confidence, although further analysis is needed.

## Evaluation Against Project Requirements

| Requirement         | Target | Result | Status            |
| ------------------- | ------ | ------ | ----------------- |
| F1 Score            | ≥ 0.75 | 0.818  | Met on validation |
| False-Positive Rate | ≤ 0.10 | 0.4148 | Not Met           |

The model currently meets the F1 score requirement but does not meet the false-positive rate requirement.

Overall, the initial results indicate that DistilRoBERTa is effective at identifying AI-generated text but is currently too likely to incorrectly flag human-written content. Reducing false positives is particularly important for RealEye because incorrectly identifying authentic content as AI-generated could mislead users.

## Saved Model

The trained model was saved to Google Drive:

`My Drive/RealEye/models/distilroberta-base-20261026-2310/`

This folder contains the full training run's saved model and associated files.

The earlier model folder ending in `-limit200` contains a small-scale training test and should not be confused with the fully trained model.

## Conclusion

The first full training run was completed successfully, establishing a working baseline for RealEye's text classifier. The model achieved an F1 score of 0.818 and recall of 97.88%, demonstrating strong detection of AI-generated text on the validation dataset.

However, its false-positive rate of 41.48% remains a significant limitation. The next priority is to improve the balance between detecting AI-generated content and correctly recognizing human-written text before deploying the classifier within RealEye.

# Classification Threshold Tuning Results

## Overview

After completing the initial DistilRoBERTa training run, additional experiments were conducted to improve the classifier's performance by adjusting its classification threshold.

The original model used a default threshold of 0.50, meaning that text was classified as AI-generated when the model's AI-class softmax score was at least 0.50.

Since the initial model had a high false-positive rate of 41.48%, different thresholds were tested to determine whether the model could reduce incorrect AI classifications while maintaining an F1 score of at least 0.75.

These experiments used the same trained model and validation dataset. **No additional training was required.**

## Threshold Tuning Results

| Classification Threshold | F1 Score  | False-Positive Rate |
| ------------------------ | --------- | ------------------- |
| 0.50 (50%)               | 0.818     | 41.48%              |
| 0.70 (70%)               | 0.827     | 37.96%              |
| 0.75 (75%)               | 0.829     | 37.00%              |
| 0.80 (80%)               | 0.830     | 36.00%              |
| 0.85 (85%)               | 0.837     | 33.90%              |
| 0.90 (90%)               | 0.841     | 31.80%              |
| 0.95 (95%)               | 0.848     | 29.10%              |
| 0.98 (98%)               | 0.854     | 25.20%              |
| 0.99 (99%)               | 0.855     | 22.30%              |
| 0.995 (99.5%)            | 0.852     | 19.00%              |
| **0.999 (99.9%)**        | **0.826** | **9.30%**           |
| **0.9995 (99.95%)**      | **0.786** | **4.30%**           |
| 0.9999 (99.99%)          | 0.000     | 0.00%               |

## Analysis of Results

Increasing the classification threshold generally reduced the false-positive rate, meaning fewer human-written passages were incorrectly identified as AI-generated.

The F1 score initially improved as the threshold increased, reaching its highest observed value of 0.855 at a threshold of 0.99. However, the false-positive rate at this threshold was still 22.30%, exceeding RealEye's target of 10%.

At a threshold of **0.999**, the model achieved an F1 score of **0.826** and a false-positive rate of **9.30%**, satisfying both project performance requirements on the validation dataset.

Increasing the threshold further to 0.9995 reduced the false-positive rate to 4.30%, but also decreased the F1 score to 0.786. Although this threshold also met both requirements, the lower F1 score indicates a greater tradeoff in overall AI-class detection performance.

At a threshold of 0.9999, the F1 score dropped to zero. This indicates that the model failed to correctly identify any AI-generated validation examples at that threshold. Although the false-positive rate was also zero, the classifier was no longer useful for detecting AI-generated content.

## Comparison With Project Requirements

| Requirement         | Target | Result at 0.999 | Status            |
| ------------------- | ------ | --------------- | ----------------- |
| F1 Score            | ≥ 0.75 | 0.826           | Met on validation |
| False-Positive Rate | ≤ 0.10 | 0.093           | Met on validation |

## Selected Candidate Threshold

Based on the validation results, **0.999 (99.9%) was selected as the primary candidate classification threshold** for further evaluation.

This threshold provides a balance between maintaining a relatively high F1 score and reducing the false-positive rate below the project's maximum acceptable value.

Compared with the original threshold of 0.50, the false-positive rate decreased from 41.48% to 9.30%, an improvement of 32.18 percentage points.

However, the threshold has not yet been finalized for deployment. The model must first be evaluated on the independent held-out test dataset to determine whether these improvements generalize to unseen examples.

Additionally, the unusually high numerical threshold suggests that the model's softmax scores may not be well calibrated. A score of 0.999 should not be interpreted as a verified 99.9% probability that the text is AI-generated.

## Next Steps

1. Evaluate the trained model using the selected 0.999 threshold on the held-out MAGE test dataset.
2. Calculate test accuracy, precision, recall, F1 score, and false-positive rate.
3. Compare test performance with validation performance to assess generalization.
4. If the test results satisfy the project requirements, integrate the model and selected threshold into RealEye's backend.
5. If performance is insufficient, investigate classification errors and consider further model improvements, like adjusting the training parameters, improving the dataset, or comparing DistilRoBERTa against other pretrained models.

## Conclusion

Threshold tuning significantly improved the performance of RealEye's initial text classifier without requiring additional training. The model achieved both project performance targets on the validation dataset at a threshold of 0.999.

These results demonstrate that adjusting the classification threshold can substantially reduce false-positive predictions while preserving useful AI-generated text detection performance. Further testing is required before confirming that the classifier is ready for integration into RealEye.
