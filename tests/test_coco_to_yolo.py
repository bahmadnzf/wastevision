from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.coco_to_yolo import CLASS_TO_ID, coco_bbox_to_yolo, convert


def test_coco_bbox_to_yolo_center_box():
    # A box exactly filling a 100x100 image should normalize to center
    # (0.5, 0.5) with full width/height (1.0, 1.0).
    xc, yc, w, h = coco_bbox_to_yolo([0, 0, 100, 100], img_w=100, img_h=100)
    assert xc == pytest.approx(0.5)
    assert yc == pytest.approx(0.5)
    assert w == pytest.approx(1.0)
    assert h == pytest.approx(1.0)


def test_coco_bbox_to_yolo_known_values():
    # 20x10 box at (10, 5) in a 200x100 image.
    xc, yc, w, h = coco_bbox_to_yolo([10, 5, 20, 10], img_w=200, img_h=100)
    assert xc == pytest.approx((10 + 10) / 200)
    assert yc == pytest.approx((5 + 5) / 100)
    assert w == pytest.approx(20 / 200)
    assert h == pytest.approx(10 / 100)


def test_coco_bbox_to_yolo_clips_out_of_bounds():
    # A box that overflows the image bounds must clip to [0, 1], not error.
    xc, yc, w, h = coco_bbox_to_yolo([90, 90, 50, 50], img_w=100, img_h=100)
    assert 0.0 <= xc <= 1.0
    assert 0.0 <= yc <= 1.0
    assert 0.0 <= w <= 1.0
    assert 0.0 <= h <= 1.0


def test_coco_bbox_to_yolo_invalid_image_dims_raises():
    with pytest.raises(ValueError):
        coco_bbox_to_yolo([0, 0, 10, 10], img_w=0, img_h=100)


def test_convert_writes_one_label_file_per_image(tmp_path: Path, synthetic_coco_dict: dict):
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(synthetic_coco_dict))
    labels_out = tmp_path / "labels"

    summary = convert(ann_path, tmp_path, labels_out)

    n_images = len(synthetic_coco_dict["images"])
    label_files = list(labels_out.glob("*.txt"))
    assert len(label_files) == n_images
    assert summary["images_written"] == n_images
    assert summary["boxes_written"] == len(synthetic_coco_dict["annotations"])


def test_convert_label_lines_have_five_fields_and_valid_class_id(tmp_path: Path, synthetic_coco_dict: dict):
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(synthetic_coco_dict))
    labels_out = tmp_path / "labels"
    convert(ann_path, tmp_path, labels_out)

    any_nonempty = False
    for label_file in labels_out.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            any_nonempty = True
            fields = line.split()
            assert len(fields) == 5
            class_id = int(fields[0])
            assert class_id in CLASS_TO_ID.values()
            for coord in fields[1:]:
                assert 0.0 <= float(coord) <= 1.0
    assert any_nonempty


def test_convert_flattens_batch_subfolder_into_stem(tmp_path: Path, synthetic_coco_dict: dict):
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(synthetic_coco_dict))
    labels_out = tmp_path / "labels"
    convert(ann_path, tmp_path, labels_out)

    # file_name "batch_1/000000.jpg" -> label "batch_1_000000.txt", not
    # nested under a batch_1/ directory.
    assert (labels_out / "batch_1_000000.txt").exists()
    assert not (labels_out / "batch_1").exists()


def test_convert_skips_unknown_category_but_keeps_going(tmp_path: Path, synthetic_coco_dict: dict):
    synthetic_coco_dict["categories"].append({"id": 999, "name": "Not A Real Category"})
    synthetic_coco_dict["annotations"].append(
        {"id": 99999, "image_id": 0, "category_id": 999, "bbox": [1, 1, 5, 5]}
    )
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(synthetic_coco_dict))
    labels_out = tmp_path / "labels"

    summary = convert(ann_path, tmp_path, labels_out)
    assert summary["skipped_unknown_category_annotations"] == 1
