"""Convert TACO's COCO-format annotations into YOLO detection labels.

TACO ships one big COCO-style `annotations.json` (all images, one file) with
segmentation polygons, not per-image YOLO `.txt` files. This module:

1. Reads the COCO json.
2. Maps each annotation's fine-grained TACO category -> one of the 8
   material groups via `src.utils.taxonomy`.
3. Converts (COCO bbox: x_min, y_min, w, h in pixels) -> (YOLO bbox:
   x_center, y_center, w, h, all normalized 0-1) for the collapsed class id.
4. Writes one `.txt` label file per image, one line per object.

Deliberately does NOT touch segmentation polygons — WasteVision trains a
detector (bounding boxes), not a segmenter, per the case study's framing.
Using boxes only also sidesteps polygon quality issues in TACO's
crowd-sourced masks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.utils.taxonomy import MATERIAL_GROUPS, get_material_group

logger = logging.getLogger(__name__)

CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(MATERIAL_GROUPS)}


def coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """[x_min, y_min, w, h] (px) -> (x_center, y_center, w, h) normalized."""
    x_min, y_min, w, h = bbox
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: {img_w}x{img_h}")
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    return (
        _clip01(x_center),
        _clip01(y_center),
        _clip01(w / img_w),
        _clip01(h / img_h),
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def convert(
    annotations_path: Path,
    images_dir: Path,
    labels_out_dir: Path,
) -> dict:
    """Convert a whole TACO annotations.json into per-image YOLO label files.

    Returns a summary dict (counts) useful for logging / sanity checks and
    for tests, so callers don't have to re-parse the filesystem to verify
    the conversion did something reasonable.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}

    labels_out_dir.mkdir(parents=True, exist_ok=True)

    # Group annotations by image so we write one file per image, including
    # images with zero annotations after category filtering (empty file —
    # YOLO treats a missing-or-empty label as "no objects", both are valid).
    anns_by_image: dict[int, list[dict]] = {img_id: [] for img_id in images}
    skipped_unknown_category = 0

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        category_name = categories[ann["category_id"]]
        try:
            material = get_material_group(category_name)
        except KeyError:
            skipped_unknown_category += 1
            logger.warning("Skipping annotation with unmapped category %r", category_name)
            continue
        anns_by_image.setdefault(img_id, []).append(
            {"class_id": CLASS_TO_ID[material.value], "bbox": ann["bbox"]}
        )

    n_images_written = 0
    n_boxes_written = 0

    for img_id, img_meta in images.items():
        file_name = img_meta["file_name"]
        img_w, img_h = img_meta["width"], img_meta["height"]
        stem = Path(file_name).stem
        # TACO file_names include a batch subfolder, e.g. "batch_1/000001.jpg"
        # — flatten with an underscore so label/image stems still match
        # 1:1 without needing to recreate the batch directory structure.
        flat_stem = stem if "/" not in file_name else file_name.replace("/", "_").rsplit(".", 1)[0]

        lines = []
        for ann in anns_by_image.get(img_id, []):
            xc, yc, w, h = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            lines.append(f"{ann['class_id']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        label_path = labels_out_dir / f"{flat_stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_images_written += 1
        n_boxes_written += len(lines)

    summary = {
        "images_written": n_images_written,
        "boxes_written": n_boxes_written,
        "skipped_unknown_category_annotations": skipped_unknown_category,
        "classes": MATERIAL_GROUPS,
    }
    logger.info("COCO->YOLO conversion summary: %s", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    convert(args.annotations, args.images_dir, args.labels_out)
