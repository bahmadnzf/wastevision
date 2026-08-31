from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.main import app, get_inference_service
from src.models.inference import InferenceService
from tests.conftest import FakeDetector


@pytest.fixture
def client():
    app.dependency_overrides[get_inference_service] = lambda: InferenceService(FakeDetector())
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _fake_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="green").save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_returns_enriched_detections(client):
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", _fake_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2  # FakeDetector's default: 2 above threshold
    assert body["count"] == len(body["detections"])
    for det in body["detections"]:
        assert det["material_group"] in {
            "plastic", "metal", "glass", "paper_cardboard",
            "organic", "textile", "composite", "other",
        }
        assert 0.0 <= det["confidence"] <= 1.0


def test_predict_rejects_unsupported_content_type(client):
    resp = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_rejects_empty_file(client):
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_predict_requires_file_field(client):
    resp = client.post("/predict")
    assert resp.status_code == 422  # FastAPI's validation error for a missing required field


def test_predict_uses_injected_fake_detector_not_real_weights(client):
    # No WASTEVISION_WEIGHTS env var or real checkpoint is set up anywhere
    # in this test process — if this call succeeds at all, the dependency
    # override is what made it possible.
    resp = client.post(
        "/predict",
        files={"file": ("test.jpg", _fake_jpeg_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
