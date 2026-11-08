"""Server settings. Each can be overridden with an environment variable."""

import os
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent

# Folder with the released text model and its calibration.json (see README: "Get the model").
TEXT_MODEL_DIR = Path(os.environ.get(
    "REALEYE_TEXT_MODEL_DIR",
    SERVER_DIR / "models" / "distilroberta-base-20261030-0247",
))

# Version of the warning rule in warning_rule.py. Change it whenever the rule or thresholds change.
RULE_VERSION = "rule-1.0"

# Text limits, in characters after whitespace normalization (SRS 5.5.3).
MIN_TEXT_CHARS = 200
MAX_TEXT_CHARS = 5000

# Tokens the text model reads, the same as in training, calibration, and testing.
MAX_TOKENS = 512

# A text counts as English when English is the detector's top language with at least this confidence.
ENGLISH_MIN_CONFIDENCE = float(os.environ.get("REALEYE_ENGLISH_MIN_CONFIDENCE", 0.5))

LIMITATION = "Not all AI-generated content contains detectable credentials or watermarks."
