"""Combines an item's separate findings into one overall state (SRS 5.1.1, 5.3.3.49-5.3.3.58).

  Warning            if any trigger: classifier estimate >= its threshold, a verified C2PA AI
                     assertion, or a positive SynthID result.
  Inconclusive       otherwise, if at least one check completed: a classifier estimate; a C2PA
                     inspection that verified a credential or found none; a SynthID check that
                     detected or did not detect a watermark.
  Unable to analyze  otherwise. Invalid credentials, unavailable, error, not applicable, and
                     unsupported results do not count as completed on their own.
"""

from .schemas import C2PAFinding, ClassifierFinding, SynthIDFinding

C2PA_COMPLETED = {"verified_ai_assertion", "verified_no_ai_assertion", "missing_credential"}
SYNTHID_COMPLETED = {"detected", "not_detected"}


def percent(value: float) -> str:
    return f"{round(value * 100)}%"


def evaluate(classifier: ClassifierFinding, c2pa: C2PAFinding, synthid: SynthIDFinding):
    """Return (overall_state, conflict, warning_evidence, explanation)."""
    classifier_done = classifier.status == "completed" and classifier.estimate is not None
    classifier_trigger = classifier_done and classifier.threshold is not None \
        and classifier.estimate >= classifier.threshold

    evidence = []
    if classifier_trigger:
        evidence.append("classifier")
    if c2pa.status == "verified_ai_assertion":
        evidence.append("c2pa")
    if synthid.status == "detected":
        evidence.append("synthid")

    completed = classifier_done or c2pa.status in C2PA_COMPLETED or synthid.status in SYNTHID_COMPLETED
    if evidence:
        state = "warning"
    elif completed:
        state = "inconclusive"
    else:
        state = "unable_to_analyze"
    conflict = bool(evidence) and classifier_done and not classifier_trigger

    return state, conflict, evidence, explain(state, classifier, evidence)


def explain(state: str, classifier: ClassifierFinding, evidence: list[str]) -> str:
    parts = []
    if classifier.status == "completed" and classifier.estimate is not None:
        side = "at or above" if "classifier" in evidence else "below"
        parts.append(f"The model estimate ({percent(classifier.estimate)}) is {side} the AI cutoff "
                     f"({percent(classifier.threshold)}).")
        if classifier.analyzed_chars and classifier.total_chars and classifier.analyzed_chars < classifier.total_chars:
            parts.append(f"This estimate is based on the first {classifier.analyzed_chars:,} of "
                         f"{classifier.total_chars:,} characters.")
    elif classifier.reason == "length":
        parts.append(f"Not analyzed: the text has {classifier.total_chars:,} characters, and RealEye "
                     "analyzes text of 200 to 5,000 characters.")
    elif classifier.reason == "language":
        parts.append("Not analyzed: the text does not appear to be English.")
    elif classifier.reason == "not_available":
        parts.append("Not analyzed: image analysis is not available yet.")
    else:
        parts.append("Not analyzed: the text model is not available on the server right now.")

    if "c2pa" in evidence:
        parts.append("Verified content credentials say this was made with AI.")
    if "synthid" in evidence:
        parts.append("A SynthID watermark was detected.")
    return " ".join(parts)
