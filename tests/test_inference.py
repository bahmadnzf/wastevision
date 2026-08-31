from __future__ import annotations

from src.models.inference import InferenceService, RawDetection
from src.utils.taxonomy import MATERIAL_GROUPS
from tests.conftest import FakeDetector


def test_predict_filters_below_confidence_threshold(fake_detector):
    service = InferenceService(fake_detector, min_confidence=0.25)
    detections = service.predict("irrelevant/path.jpg")

    # FakeDetector's default fixture has one detection at confidence 0.10,
    # which must be dropped by the 0.25 threshold.
    assert all(d.confidence >= 0.25 for d in detections)
    assert len(detections) == 2


def test_predict_enriches_with_correct_material_and_policy(fake_detector):
    service = InferenceService(fake_detector, min_confidence=0.25)
    detections = service.predict("irrelevant/path.jpg")

    plastic_det = next(d for d in detections if d.material_group == "plastic")
    assert plastic_det.confidence == 0.91
    assert plastic_det.recyclable in ("recyclable", "not_recyclable", "check_local_program")
    assert isinstance(plastic_det.special_handling, bool)
    assert plastic_det.policy_note  # non-empty


def test_predict_passes_image_path_through_to_detector(fake_detector):
    service = InferenceService(fake_detector)
    service.predict("some/specific/path.jpg")
    assert fake_detector.calls == ["some/specific/path.jpg"]


def test_predict_ignores_out_of_range_class_id():
    bad_detections = [RawDetection(class_id=999, confidence=0.9, xyxy=(0, 0, 10, 10))]
    service = InferenceService(FakeDetector(bad_detections))
    assert service.predict("x.jpg") == []


def test_predict_empty_detections_returns_empty_list():
    service = InferenceService(FakeDetector([]))
    assert service.predict("x.jpg") == []


def test_detection_to_dict_round_trips_all_fields(fake_detector):
    service = InferenceService(fake_detector, min_confidence=0.0)
    detections = service.predict("x.jpg")
    d = detections[0].to_dict()
    assert set(d.keys()) == {
        "material_group",
        "confidence",
        "bbox_xyxy",
        "recyclable",
        "special_handling",
        "policy_note",
    }


def test_material_group_of_every_class_id_is_valid():
    # class_id N must always enrich to MATERIAL_GROUPS[N] — this is the
    # contract the training pipeline's CLASS_TO_ID mapping depends on.
    for class_id, name in enumerate(MATERIAL_GROUPS):
        service = InferenceService(
            FakeDetector([RawDetection(class_id=class_id, confidence=1.0, xyxy=(0, 0, 1, 1))])
        )
        detections = service.predict("x.jpg")
        assert detections[0].material_group == name
