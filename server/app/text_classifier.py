"""Runs the released text model exactly as it was trained, calibrated, and tested.

Pipeline (matches ml/text/analyze_errors.py and ml/text/calibrate.py):
  tokenize (first 512 tokens) -> model logits -> margin = logit[AI] - logit[human]
  -> Platt calibration: estimate = sigmoid(a * margin + b) -> "ai" if estimate >= cutoff
"""

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import MAX_TOKENS

log = logging.getLogger("realeye")


@dataclass
class TextPrediction:
    estimate: float  # calibrated AI likelihood, 0 to 1
    label: str  # "ai" or "human"
    analyzed_chars: int  # characters actually read by the model (first 512 tokens)
    total_chars: int


class TextClassifier:
    def __init__(self, model_dir: Path):
        calibration = json.loads((model_dir / "calibration.json").read_text())
        if calibration.get("method") != "platt":
            raise ValueError(f"{model_dir}/calibration.json must use Platt scaling (method: platt)")
        self.a = float(calibration["a"])
        self.b = float(calibration["b"])
        self.cutoff = float(calibration["cutoff"])
        self.version = model_dir.name
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()

    @classmethod
    def try_load(cls, model_dir: Path) -> "TextClassifier | None":
        """Load the model, or return None (and log why) so the server can still start."""
        try:
            classifier = cls(model_dir)
        except Exception as error:  # missing folder, missing file, bad calibration
            log.warning("Text model not loaded from %s: %s", model_dir, error)
            return None
        log.info("Loaded text model %s (cutoff %.4f)", classifier.version, classifier.cutoff)
        return classifier

    def predict(self, texts: list[str], batch_size: int = 16) -> list[TextPrediction]:
        predictions = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = self.tokenizer(batch_texts, truncation=True, max_length=MAX_TOKENS, padding=True,
                                     return_tensors="pt", return_offsets_mapping=True,
                                     return_special_tokens_mask=True)
            offsets = encoded.pop("offset_mapping")
            special = encoded.pop("special_tokens_mask")
            with torch.no_grad():
                logits = self.model(**encoded).logits.float()
            margins = (logits[:, 1] - logits[:, 0]).tolist()
            for i, text in enumerate(batch_texts):
                ends = [int(end) for (_, end), is_special in zip(offsets[i].tolist(), special[i].tolist())
                        if not is_special]
                estimate = 1 / (1 + math.exp(-(self.a * margins[i] + self.b)))
                predictions.append(TextPrediction(
                    estimate=estimate,
                    label="ai" if estimate >= self.cutoff else "human",
                    analyzed_chars=max(ends, default=0),
                    total_chars=len(text),
                ))
        return predictions
