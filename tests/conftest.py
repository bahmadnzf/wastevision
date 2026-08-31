"""Shared fixtures: a synthetic COCO-format annotations file (small, valid
TACO category names, deliberately imbalanced across images) plus a fake
Detector so the whole test suite runs against synthetic data / mocked
models — no GPU, no real weights, no ultralytics import required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models.inference import RawDetection

# Real TACO category names, spanning several material groups, used to build
# a small synthetic dataset with a deliberately skewed distribution: lots
# of "Cigarette" (-> other), a few "Glass bottle" (-> glass), one "Food
# waste" (-> organic) — enough imbalance to make stratification tests
# meaningful.
_CATEGORY_NAMES = [
    "Cigarette",           # -> other
    "Clear plastic bottle",  # -> plastic
    "Glass bottle",         # -> glass
    "Food waste",           # -> organic
    "Aluminium foil",       # -> metal
]


def _make_coco_dict(n_images: int = 20, seed: int = 0) -> dict:
    import random

    rng = random.Random(seed)
    categories = [{"id": i, "name": name} for i, name in enumerate(_CATEGORY_NAMES)]
    images = []
    annotations = []
    ann_id = 0

    for img_id in range(n_images):
        w, h = 800, 600
        images.append(
            {
                "id": img_id,
                "file_name": f"batch_{img_id % 3 + 1}/{img_id:06d}.jpg",
                "width": w,
                "height": h,
            }
        )
        # Cigarette-heavy distribution: most images get 2-3 cigarette
        # annotations, a minority get one rarer-class annotation instead.
        if img_id % 4 == 0:
            cat_id = rng.choice([2, 3, 4])  # glass / organic / metal minority
            n_objs = 1
        else:
            cat_id = 0  # cigarette majority
            n_objs = rng.randint(1, 3)

        for _ in range(n_objs):
            bw, bh = rng.randint(20, 100), rng.randint(20, 100)
            x = rng.randint(0, w - bw)
            y = rng.randint(0, h - bh)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": [x, y, bw, bh],
                }
            )
            ann_id += 1

    # One deliberately image with zero annotations, to exercise the
    # "unlabeled" bucket in split.py.
    images.append({"id": n_images, "file_name": f"batch_1/{n_images:06d}.jpg", "width": 800, "height": 600})

    return {"images": images, "annotations": annotations, "categories": categories}


@pytest.fixture
def synthetic_coco_dict() -> dict:
    return _make_coco_dict()


@pytest.fixture
def synthetic_coco_path(tmp_path: Path, synthetic_coco_dict: dict) -> Path:
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(synthetic_coco_dict))
    return path


class FakeDetector:
    """Satisfies the `Detector` protocol without ultralytics or real
    weights. Returns a fixed, scripted set of raw detections regardless of
    the image path, so tests can assert on exact enrichment output."""

    def __init__(self, detections: list[RawDetection] | None = None):
        self._detections = detections if detections is not None else [
            RawDetection(class_id=0, confidence=0.91, xyxy=(10.0, 10.0, 100.0, 100.0)),  # plastic
            RawDetection(class_id=1, confidence=0.55, xyxy=(50.0, 50.0, 150.0, 150.0)),  # metal
            RawDetection(class_id=0, confidence=0.10, xyxy=(0.0, 0.0, 20.0, 20.0)),      # plastic, below default threshold
        ]
        self.calls: list[str] = []

    def predict_raw(self, image_path):
        self.calls.append(str(image_path))
        return self._detections


@pytest.fixture
def fake_detector() -> FakeDetector:
    return FakeDetector()
