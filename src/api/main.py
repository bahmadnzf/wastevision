"""FastAPI service for WasteVision.

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

`get_inference_service` is a FastAPI dependency, not a hardcoded global —
this is what lets the test suite override it with a fake detector via
`app.dependency_overrides[get_inference_service] = ...` and fully exercise
the API contract (routing, validation, response schema, error handling)
without ever loading real weights or importing ultralytics.
"""

from __future__ import annotations

import logging
import os
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from src.api.schemas import DetectionOut, HealthResponse, PredictResponse
from src.models.inference import InferenceService, load_inference_service

logger = logging.getLogger(__name__)

app = FastAPI(
    title="WasteVision API",
    description=(
        "Detects waste items in an image, classifies each item's material "
        "group, and applies a recyclability policy. See CASE_STUDY.md for "
        "the perception/policy design rationale."
    ),
    version="0.1.0",
)

WEIGHTS_ENV_VAR = "WASTEVISION_WEIGHTS"
DEFAULT_WEIGHTS_PATH = "/app/models/best.pt"
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@lru_cache(maxsize=1)
def _load_default_service() -> InferenceService:
    weights_path = os.environ.get(WEIGHTS_ENV_VAR, DEFAULT_WEIGHTS_PATH)
    if not Path(weights_path).exists():
        raise RuntimeError(
            f"No weights found at {weights_path!r}. Set the {WEIGHTS_ENV_VAR} "
            "env var to a trained checkpoint, or train one via "
            "`python -m src.models.train`."
        )
    return load_inference_service(weights_path)


def get_inference_service() -> InferenceService:
    return _load_default_service()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service),
) -> PredictResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type {file.content_type!r}. "
                f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
            ),
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(contents)
        tmp.flush()
        try:
            detections = service.predict(tmp.name)
        except Exception as exc:  # pragma: no cover - defensive catch-all
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed.") from exc

    return PredictResponse(
        detections=[DetectionOut(**d.to_dict()) for d in detections],
        count=len(detections),
    )
