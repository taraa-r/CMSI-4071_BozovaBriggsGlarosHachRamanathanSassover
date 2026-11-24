
"""Combine detector findings into an overall RealEye result."""

from .schemas import (
    C2PAFinding,
    ClassifierFinding,
    SynthIDFinding,
)

C2PA_COMPLETED = {
    "verified_ai_assertion",
    "verified_no_ai_assertion",
    "missing_credential",
}

SYNTHID_COMPLETED = {
    "detected",
    "not_detected",
}


def percent(value: float) -> str:
    return f"{round(value * 100)}%"


def evaluate(
    classifier: ClassifierFinding,
    c2pa: C2PAFinding,
    synthid: SynthIDFinding,
):
    """Return state, conflict, evidence, and explanation."""

    classifier_done = (
        classifier.status == "completed"
        and classifier.estimate is not None
    )

    classifier_trigger = (
        classifier_done
        and classifier.threshold is not None
        and classifier.estimate >= classifier.threshold
    )

    evidence = []

    if classifier_trigger:
        evidence.append("classifier")

    if c2pa.status == "verified_ai_assertion":
        evidence.append("c2pa")

    if synthid.status == "detected":
        evidence.append("synthid")

    completed = (
        classifier_done
        or c2pa.status in C2PA_COMPLETED
        or synthid.status in SYNTHID_COMPLETED
    )

    if evidence:
        state = "warning"
    elif completed:
        state = "inconclusive"
    else:
        state = "unable_to_analyze"

    conflict = (
        bool(evidence)
        and classifier_done
        and not classifier_trigger
    )

    explanation = explain(
        state,
        classifier,
        c2pa,
        synthid,
        evidence,
    )

    return state, conflict, evidence, explanation


def explain(
    state: str,
    classifier: ClassifierFinding,
    c2pa: C2PAFinding,
    synthid: SynthIDFinding,
    evidence: list[str],
) -> str:

    parts = []

    if classifier.reason == "not_available":
        # Image item: the text classifier does not apply.
        if c2pa.status == "missing_credential":
            parts.append(
                "No C2PA Content Credentials were found "
                "in this image."
            )
        elif c2pa.status == "invalid_credential":
            parts.append(
                "The image contains C2PA credentials "
                "that failed validation."
            )
        elif c2pa.status == "unavailable":
            parts.append(
                "C2PA inspection could not establish "
                "trusted provenance for this image."
            )
        elif c2pa.status == "error":
            parts.append(
                "C2PA inspection encountered an error."
            )

        if synthid.status == "unavailable":
            parts.append(
                "SynthID detection is not currently available."
            )
        elif synthid.status == "error":
            parts.append(
                "SynthID detection encountered an error."
            )
        elif synthid.status == "not_detected":
            parts.append(
                "No SynthID watermark was detected."
            )

    elif (
        classifier.status == "completed"
        and classifier.estimate is not None
    ):
        side = (
            "at or above"
            if "classifier" in evidence
            else "below"
        )

        parts.append(
            f"The model estimate "
            f"({percent(classifier.estimate)}) is "
            f"{side} the AI cutoff "
            f"({percent(classifier.threshold)})."
        )

        if (
            classifier.analyzed_chars
            and classifier.total_chars
            and classifier.analyzed_chars
            < classifier.total_chars
        ):
            parts.append(
                f"This estimate is based on the first "
                f"{classifier.analyzed_chars:,} of "
                f"{classifier.total_chars:,} characters."
            )

    elif classifier.reason == "length":
        parts.append(
            f"Not analyzed: the text has "
            f"{classifier.total_chars:,} characters, "
            "and RealEye analyzes text of "
            "200 to 5,000 characters."
        )

    elif classifier.reason == "language":
        parts.append(
            "Not analyzed: the text does not "
            "appear to be English."
        )

    else:
        parts.append(
            "Not analyzed: the text model is "
            "not available on the server right now."
        )

    if "c2pa" in evidence:
        parts.append(
            "Verified content credentials indicate "
            "AI-generated content."
        )

    if "synthid" in evidence:
        parts.append(
            "A SynthID watermark was detected."
        )

    return " ".join(parts)
