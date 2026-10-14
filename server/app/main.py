from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import (
    AnalyzeItem,
    AnalyzeRequest,
    AnalyzeResponse,
    C2PAFinding,
    ClassifierFinding,
    ItemResult,
    SynthIDFinding,
)

app = FastAPI(title="RealEye API")

RULE_VERSION = "fake-0.1"
LIMITATION = "Not all AI-generated content carries credentials or watermarks."


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
    return {"status": "ok"}


def fake_result(item: AnalyzeItem) -> ItemResult:
    # FAKE: no model, C2PA, or SynthID yet. Always a below-threshold estimate,
    # so every item comes back "inconclusive". Replace with real checks later.
    return ItemResult(
        item_id=item.item_id,
        type=item.type,
        overall_state="inconclusive",
        conflict=False,
        warning_evidence=[],
        classifier=ClassifierFinding(
            status="completed",
            estimate=0.5,
            label="human",
            threshold=0.7,
            model_version=f"fake-{item.type}-0.1",
        ),
        c2pa=C2PAFinding(status="not_applicable" if item.type == "text" else "unavailable"),
        synthid=SynthIDFinding(status="unavailable"),
        limitations=[LIMITATION],
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest):
    if not body.items:
        return error_response(400, "no_items", "The request contains no items to analyze.")

    return AnalyzeResponse(
        scan_id=body.scan_id,
        rule_version=RULE_VERSION,
        results=[fake_result(item) for item in body.items],
    )
