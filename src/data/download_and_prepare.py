"""One-command TACO -> YOLO dataset pipeline.

TACO's images are Flickr-hosted and fetched via TACO's own downloader (see
README) — this script picks up *after* that, starting from a local
`images_dir` containing the downloaded batches and TACO's `annotations.json`.

Pipeline:
    1. Split image ids into train/val/test (stratified, leak-free — see
       `src.data.split`).
    2. For each split, copy the source images into
       `<output-dir>/<split>/images/` and write matching YOLO label files
       into `<output-dir>/<split>/labels/` (60 TACO categories collapsed to
       8 material groups — see `src.utils.taxonomy`).
    3. Write `<output-dir>/data.yaml`, ready to hand to
       `ultralytics.YOLO(...).train(data=...)`.

Usage:
    python -m src.data.download_and_prepare \\
        --images-dir TACO/data --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from src.data.coco_to_yolo import coco_bbox_to_yolo
from src.data.split import split_from_coco
from src.utils.taxonomy import MATERIAL_GROUPS, get_material_group

logger = logging.getLogger(__name__)

CLASS_TO_ID = {name: idx for idx, name in enumerate(MATERIAL_GROUPS)}


def _flat_stem(file_name: str) -> str:
    """TACO file names look like 'batch_1/000001.jpg'; flatten the batch
    folder into the stem so image/label files stay 1:1 without recreating
    TACO's nested directory layout under data/processed."""
    if "/" not in file_name:
        return Path(file_name).stem
    return file_name.replace("/", "_").rsplit(".", 1)[0]


def prepare(
    annotations_path: Path,
    images_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    with open(annotations_path) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    splits = split_from_coco(annotations_path, train_ratio, val_ratio, test_ratio, seed)

    summary: dict[str, dict] = {}
    for split_name, image_records in splits.items():
        img_out = output_dir / split_name / "images"
        lbl_out = output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        n_copied, n_missing, n_boxes = 0, 0, 0
        for rec in image_records:
            src = images_dir / rec["file_name"]
            stem = _flat_stem(rec["file_name"])
            if not src.exists():
                n_missing += 1
                logger.warning("Missing source image, skipping: %s", src)
                continue

            shutil.copy2(src, img_out / f"{stem}{src.suffix}")

            lines = []
            for ann in anns_by_image.get(rec["id"], []):
                category_name = categories[ann["category_id"]]
                try:
                    material = get_material_group(category_name)
                except KeyError:
                    continue
                xc, yc, w, h = coco_bbox_to_yolo(ann["bbox"], rec["width"], rec["height"])
                lines.append(f"{CLASS_TO_ID[material.value]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_copied += 1
            n_boxes += len(lines)

        summary[split_name] = {"images": n_copied, "missing": n_missing, "boxes": n_boxes}
        logger.info("%s: %s", split_name, summary[split_name])

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {i: name for i, name in enumerate(MATERIAL_GROUPS)},
    }
    with open(output_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    logger.info("Wrote %s", output_dir / "data.yaml")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=None, help="Defaults to <images-dir>/annotations.json")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    ann_path = args.annotations or (args.images_dir / "annotations.json")
    prepare(
        ann_path,
        args.images_dir,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
