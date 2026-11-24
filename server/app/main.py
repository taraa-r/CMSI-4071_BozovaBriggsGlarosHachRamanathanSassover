
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .detector_runner import run_image_detectors
from .image_validation import validate_image
from .language import LanguageChecker
from .rate_limit import check_rate_limit
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

MAX_REQUEST_BYTES = 16 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.text_classifier = TextClassifier.try_load(
        config.TEXT_MODEL_DIR
    )
    app.state.language = LanguageChecker()
    yield


app = FastAPI(title="RealEye API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_origin_regex=r"chrome-extension://[a-p]{32}",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def error_response(
    status: int, code: str, message: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


class RequestBodyTooLarge(Exception):
    pass


@app.middleware("http")
async def request_size_limit(request: Request, call_next):
    """Reject POST /analyze bodies larger than 16 MiB."""
    if request.url.path != "/analyze" or request.method != "POST":
        return await call_next(request)

    content_length = request.headers.get("content-length")

    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return error_response(
                    413,
                    "request_too_large",
                    "Request body exceeds the 16 MiB limit.",
                )
        except ValueError:
            return error_response(
                400,
                "invalid_request",
                "Invalid Content-Length header.",
            )

    received = 0
    original_receive = request.receive

    async def limited_receive():
        nonlocal received
        message = await original_receive()

        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > MAX_REQUEST_BYTES:
                raise RequestBodyTooLarge()

        return message

    request._receive = limited_receive

    try:
        return await call_next(request)
    except RequestBodyTooLarge:
        return error_response(
            413,
            "request_too_large",
            "Request body exceeds the 16 MiB limit.",
        )


@app.exception_handler(RequestBodyTooLarge)
async def request_body_too_large(
    request: Request, exc: RequestBodyTooLarge
):
    return error_response(
        413,
        "request_too_large",
        "Request body exceeds the 16 MiB limit.",
    )


@app.exception_handler(RequestValidationError)
def validation_error(
    request: Request, exc: RequestValidationError
):
    first = exc.errors()[0]
    field = ".".join(
        str(part) for part in first["loc"] if part != "body"
    )
    return error_response(
        422,
        "invalid_request",
        f"{field}: {first['msg']}",
    )


@app.exception_handler(Exception)
def unexpected_error(_request: Request, _exc: Exception):
    _ = (_request, _exc)
    logging.getLogger("realeye").exception("Unhandled API error")
    return error_response(500, "server_error", "The RealEye server encountered an error.")


@app.get("/health")
def health():
    classifier = app.state.text_classifier
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "model_version": (
            classifier.version if classifier else None
        ),
        "rule_version": config.RULE_VERSION,
    }


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def classify_texts(items: list[AnalyzeItem]) -> dict:
    classifier = app.state.text_classifier
    findings, to_score = {}, []

    for item in items:
        text = normalize(item.text or "")

        if not (
            config.MIN_TEXT_CHARS
            <= len(text)
            <= config.MAX_TEXT_CHARS
        ):
            findings[item.item_id] = ClassifierFinding(
                status="unsupported",
                reason="length",
                total_chars=len(text),
            )
        elif not app.state.language.is_english(text):
            findings[item.item_id] = ClassifierFinding(
                status="unsupported",
                reason="language",
                total_chars=len(text),
            )
        elif classifier is None:
            findings[item.item_id] = ClassifierFinding(
                status="error"
            )
        else:
            to_score.append((item.item_id, text))

    if to_score:
        try:
            predictions = classifier.predict(
                [text for _, text in to_score]
            )
        except Exception:
            logging.getLogger("realeye").exception(
                "Text inference failed"
            )
            predictions = None

        for i, (item_id, _) in enumerate(to_score):
            if predictions is None:
                findings[item_id] = ClassifierFinding(
                    status="error"
                )
                continue

            p = predictions[i]
            findings[item_id] = ClassifierFinding(
                status="completed",
                estimate=p.estimate,
                label=p.label,
                threshold=classifier.cutoff,
                model_version=classifier.version,
                analyzed_chars=p.analyzed_chars,
                total_chars=p.total_chars,
            )

    return findings


def item_result(
    item: AnalyzeItem,
    classifier: ClassifierFinding,
    image_error: str | None = None,
    c2pa_status: str | None = None,
    synthid_status: str | None = None,
) -> ItemResult:

    if item.type == "text":
        c2pa = C2PAFinding(status="not_applicable")
        synthid = SynthIDFinding(status="not_applicable")
    else:
        c2pa = C2PAFinding(
            status=c2pa_status or "unavailable"
        )
        synthid = SynthIDFinding(
            status=synthid_status or "unavailable"
        )

    state, conflict, evidence, explanation = evaluate(
        classifier, c2pa, synthid
    )

    limitations = [config.LIMITATION]

    if image_error is not None:
        state = "unable_to_analyze"
        conflict = False
        evidence = []
        explanation = f"Unsupported image: {image_error}"
        limitations.append(image_error)

    return ItemResult(
        item_id=item.item_id,
        type=item.type,
        overall_state=state,
        conflict=conflict,
        warning_evidence=evidence,
        classifier=classifier,
        c2pa=c2pa,
        synthid=synthid,
        limitations=limitations,
        explanation=explanation,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest, request: Request):

    # Check the request limit before running detectors.
    rate_limit_response = check_rate_limit(request)

    if rate_limit_response is not None:
        return rate_limit_response

    if not body.items:
        return error_response(
            400,
            "no_items",
            "The request contains no items to analyze.",
        )

    text_findings = classify_texts(
        [
            item for item in body.items
            if item.type == "text"
        ]
    )

    results = []

    # Track images that pass validation.
    valid_image_count = 0

    for item in body.items:
        if item.type == "text":
            classifier = text_findings[item.item_id]
            results.append(
                item_result(item, classifier)
            )
            continue

        # Validate the image before running detectors.
        image_bytes, image_error = validate_image(
            item.image_base64,
            item.mime_type,
        )

        classifier = ClassifierFinding(
            status="unsupported",
            reason="not_available",
        )

        if image_error is not None:
            results.append(
                item_result(
                    item,
                    classifier,
                    image_error=image_error,
                )
            )
            continue

        # This image passed validation.
        valid_image_count += 1

        # Run C2PA and SynthID checks concurrently.
        c2pa_status, synthid_status = (
            run_image_detectors(
                image_bytes,
                item.mime_type,
            )
        )

        results.append(
            item_result(
                item,
                classifier,
                c2pa_status=c2pa_status,
                synthid_status=synthid_status,
            )
        )

    # A valid image can be processed even when detectors
    # return unavailable or inconclusive.
    #
    # Text items count only when classification completed.
    analyzable = (
        valid_image_count > 0
        or any(
            result.classifier.status == "completed"
            for result in results
        )
    )

    if not analyzable:
        return error_response(
            400,
            "no_analyzable_items",
            "None of the submitted items could be analyzed.",
        )

    return AnalyzeResponse(
        scan_id=body.scan_id,
        rule_version=config.RULE_VERSION,
        results=results,
    )
