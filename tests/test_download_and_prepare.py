"""End-to-end test of the whole COCO -> YOLO dataset prep pipeline,
including data.yaml generation — the closest thing to an integration test
that still needs no GPU, no real TACO download, and no network access."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

from src.data.download_and_prepare import prepare
from src.utils.taxonomy import MATERIAL_GROUPS


def _write_dummy_images(images_dir: Path, coco_dict: dict) -> None:
    for img in coco_dict["images"]:
        path = images_dir / img["file_name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (img["width"], img["height"]), color="white").save(path)


def test_prepare_end_to_end(tmp_path: Path, synthetic_coco_dict: dict):
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(synthetic_coco_dict))

    images_dir = tmp_path / "raw_images"
    _write_dummy_images(images_dir, synthetic_coco_dict)

    output_dir = tmp_path / "processed"
    summary = prepare(ann_path, images_dir, output_dir, seed=42)

    # Every split directory exists with matching images/labels counts.
    for split_name in ("train", "val", "test"):
        img_files = list((output_dir / split_name / "images").glob("*"))
        lbl_files = list((output_dir / split_name / "labels").glob("*.txt"))
        assert len(img_files) == len(lbl_files) == summary[split_name]["images"]

    total_images = sum(s["images"] for s in summary.values())
    assert total_images == len(synthetic_coco_dict["images"])

    # data.yaml is valid, points at the right dirs, and lists all 8 classes
    # in the exact order training/inference depend on.
    data_yaml = yaml.safe_load((output_dir / "data.yaml").read_text())
    assert data_yaml["train"] == "train/images"
    assert data_yaml["val"] == "val/images"
    assert data_yaml["test"] == "test/images"
    assert [data_yaml["names"][i] for i in range(8)] == MATERIAL_GROUPS


def test_prepare_reports_missing_source_images(tmp_path: Path, synthetic_coco_dict: dict):
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(synthetic_coco_dict))

    images_dir = tmp_path / "raw_images"  # deliberately never populated
    images_dir.mkdir()

    output_dir = tmp_path / "processed"
    summary = prepare(ann_path, images_dir, output_dir, seed=42)

    total_missing = sum(s["missing"] for s in summary.values())
    total_copied = sum(s["images"] for s in summary.values())
    assert total_missing == len(synthetic_coco_dict["images"])
    assert total_copied == 0
