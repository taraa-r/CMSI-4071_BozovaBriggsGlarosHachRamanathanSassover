"""Request and response shapes for POST /analyze. See docs/api-contract.md."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

MAX_ITEMS_PER_REQUEST = 10  # SRS 5.3.3.69


# ---------- Request ----------

class AnalyzeItem(BaseModel):
    item_id: UUID
    type: Literal["text", "image"]
    text: str | None = None
    image_base64: str | None = None
    mime_type: Literal["image/jpeg", "image/png", "image/webp"] | None = None
    source_url: str | None = None


class AnalyzeRequest(BaseModel):
    scan_id: UUID
    page_url: str
    scan_time: datetime
    items: list[AnalyzeItem] = Field(max_length=MAX_ITEMS_PER_REQUEST)


# ---------- Response ----------

OverallState = Literal["warning", "inconclusive", "unable_to_analyze"]


class ClassifierFinding(BaseModel):
    status: Literal["completed", "unsupported", "error"]
    estimate: float | None = None
    estimate_kind: Literal["model_estimate"] = "model_estimate"
    label: Literal["ai", "human"] | None = None
    threshold: float | None = None
    model_version: str | None = None


class C2PAFinding(BaseModel):
    status: Literal[
        "verified_ai_assertion",
        "verified_no_ai_assertion",
        "missing_credential",
        "invalid_credential",
        "not_applicable",
        "unavailable",
        "error",
    ]


class SynthIDFinding(BaseModel):
    status: Literal["detected", "not_detected", "not_applicable", "unavailable", "error"]


class ItemResult(BaseModel):
    item_id: UUID
    type: Literal["text", "image"]
    overall_state: OverallState
    conflict: bool
    warning_evidence: list[Literal["classifier", "c2pa", "synthid"]]
    classifier: ClassifierFinding
    c2pa: C2PAFinding
    synthid: SynthIDFinding
    limitations: list[str]


class AnalyzeResponse(BaseModel):
    scan_id: UUID
    rule_version: str
    results: list[ItemResult]
