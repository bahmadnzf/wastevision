"""Inference layer: raw detector output -> taxonomy + recyclability policy.

Deliberately split into two pieces:

- `Detector` (a Protocol) / `YoloDetector` (the real ultralytics adapter) —
  the only place that imports `ultralytics` or touches actual model weights.
- `InferenceService` — pure enrichment logic (class id -> material group ->
  recyclability verdict). It depends on the `Detector` protocol, not on
  ultralytics, so unit tests can inject a fake detector and fully exercise
  the enrichment logic without a GPU, real weights, or ultralytics even
  being importable. FastAPI's dependency-override system does the same
  injection at the API layer (see src/api/main.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from src.utils.taxonomy import MATERIAL_GROUPS, get_verdict

BBox = tuple[float, float, float, float]  # x_min, y_min, x_max, y_max in pixels


@dataclass(frozen=True)
class RawDetection:
    """One detection straight from the model, before enrichment."""

    class_id: int
    confidence: float
    xyxy: BBox


@runtime_checkable
class Detector(Protocol):
    """Minimal interface InferenceService needs from an underlying model.

    Any object implementing `predict_raw` can be injected — a real YOLO
    wrapper for production, a scripted fake for tests, or in principle a
    different architecture entirely.
    """

    def predict_raw(self, image_path: str | Path) -> Sequence[RawDetection]: ...


@dataclass(frozen=True)
class Detection:
    """A detection enriched with material group + recyclability policy —
    this is the shape returned to API/UI callers."""

    material_group: str
    confidence: float
    bbox_xyxy: BBox
    recyclable: str
    special_handling: bool
    policy_note: str

    def to_dict(self) -> dict:
        return asdict(self)


class YoloDetector:
    """Adapter around ultralytics.YOLO satisfying the `Detector` protocol.

    `ultralytics` is imported lazily inside __init__, not at module level,
    so importing this module (or anything that imports it, like the API
    schemas) never requires ultralytics to be installed unless you actually
    instantiate this class.
    """

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        from ultralytics import YOLO  # noqa: PLC0415 — intentional lazy import

        self._model = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def predict_raw(self, image_path: str | Path) -> list[RawDetection]:
        results = self._model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        raw: list[RawDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                raw.append(
                    RawDetection(
                        class_id=int(box.cls.item()),
                        confidence=float(box.conf.item()),
                        xyxy=tuple(box.xyxy[0].tolist()),
                    )
                )
        return raw


class InferenceService:
    """Runs a Detector, filters by confidence, attaches material +
    recyclability info. This is what the API route and the Gradio app both
    call — neither talks to YoloDetector or ultralytics directly."""

    def __init__(self, detector: Detector, min_confidence: float = 0.25):
        self._detector = detector
        self.min_confidence = min_confidence

    def predict(self, image_path: str | Path) -> list[Detection]:
        detections: list[Detection] = []
        for raw in self._detector.predict_raw(image_path):
            if raw.confidence < self.min_confidence:
                continue
            if not (0 <= raw.class_id < len(MATERIAL_GROUPS)):
                # Defensive: a mismatched weights/taxonomy version could
                # otherwise produce a silent IndexError deep in a request.
                continue
            material = MATERIAL_GROUPS[raw.class_id]
            policy = get_verdict(material)
            detections.append(
                Detection(
                    material_group=material,
                    confidence=round(raw.confidence, 4),
                    bbox_xyxy=raw.xyxy,
                    recyclable=policy.verdict.value,
                    special_handling=policy.special_handling,
                    policy_note=policy.note,
                )
            )
        return detections


def load_inference_service(
    weights_path: str | Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    device: str = "cpu",
    min_confidence: float = 0.25,
) -> InferenceService:
    """Convenience factory wiring a real YoloDetector into InferenceService.
    Used by the API/Gradio entrypoints; tests build InferenceService
    directly with a fake Detector instead of calling this."""
    detector = YoloDetector(weights_path, conf_threshold, iou_threshold, device)
    return InferenceService(detector, min_confidence)
