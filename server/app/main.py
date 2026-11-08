import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import config
from .language import LanguageChecker
from .schemas import (
    AnalyzeItem,
    AnalyzeRequest,
    AnalyzeResponse,
    C2PAFinding,
    ClassifierFinding,
    ItemResult,
    SynthIDFinding,
)
from .text_classifier import TextClassifier
from .warning_rule import evaluate

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup. If the model folder is missing, the server still runs and text items
    # come back "unable_to_analyze"; /health shows model_loaded: false.
    app.state.text_classifier = TextClassifier.try_load(config.TEXT_MODEL_DIR)
    app.state.language = LanguageChecker()
    yield


app = FastAPI(title="RealEye API", lifespan=lifespan)


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(RequestValidationError)
def validation_error(request: Request, exc: RequestValidationError):
    # Return 422 in the contract's error format instead of FastAPI's default shape.
    first = exc.errors()[0]
    field = ".".join(str(part) for part in first["loc"] if part != "body")
    return error_response(422, "invalid_request", f"{field}: {first['msg']}")


@app.get("/health")
def health():
    classifier = app.state.text_classifier
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "model_version": classifier.version if classifier else None,
        "rule_version": config.RULE_VERSION,
    }


def normalize(text: str) -> str:
    """Collapse whitespace the same way the training data was cleaned (ml/text/mage.py)."""
    return re.sub(r"\s+", " ", text).strip()


def classify_texts(items: list[AnalyzeItem]) -> dict:
    """Classifier finding for each text item, keyed by item_id."""
    classifier = app.state.text_classifier
    findings, to_score = {}, []
    for item in items:
        text = normalize(item.text or "")
        if not config.MIN_TEXT_CHARS <= len(text) <= config.MAX_TEXT_CHARS:
            findings[item.item_id] = ClassifierFinding(status="unsupported", reason="length", total_chars=len(text))
        elif not app.state.language.is_english(text):
            findings[item.item_id] = ClassifierFinding(status="unsupported", reason="language", total_chars=len(text))
        elif classifier is None:
            findings[item.item_id] = ClassifierFinding(status="error")
        else:
            to_score.append((item.item_id, text))

    if to_score:
        try:
            predictions = classifier.predict([text for _, text in to_score])
        except Exception:
            logging.getLogger("realeye").exception("Text inference failed")
            predictions = None
        for i, (item_id, _) in enumerate(to_score):
            if predictions is None:
                findings[item_id] = ClassifierFinding(status="error")
                continue
            p = predictions[i]
            findings[item_id] = ClassifierFinding(
                status="completed", estimate=p.estimate, label=p.label, threshold=classifier.cutoff,
                model_version=classifier.version, analyzed_chars=p.analyzed_chars, total_chars=p.total_chars,
            )
    return findings


def item_result(item: AnalyzeItem, classifier: ClassifierFinding) -> ItemResult:
    # C2PA applies only to images; SynthID has no authorized detector yet (SRS 5.5.1).
    c2pa = C2PAFinding(status="not_applicable" if item.type == "text" else "unavailable")
    synthid = SynthIDFinding(status="unavailable")
    state, conflict, evidence, explanation = evaluate(classifier, c2pa, synthid)
    return ItemResult(
        item_id=item.item_id, type=item.type, overall_state=state, conflict=conflict,
        warning_evidence=evidence, classifier=classifier, c2pa=c2pa, synthid=synthid,
        limitations=[config.LIMITATION], explanation=explanation,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest):
    if not body.items:
        return error_response(400, "no_items", "The request contains no items to analyze.")

    text_findings = classify_texts([item for item in body.items if item.type == "text"])
    results = []
    for item in body.items:
        if item.type == "text":
            classifier = text_findings[item.item_id]
        else:
            # The image classifier is not released yet.
            classifier = ClassifierFinding(status="unsupported", reason="not_available")
        results.append(item_result(item, classifier))

    return AnalyzeResponse(scan_id=body.scan_id, rule_version=config.RULE_VERSION, results=results)
