"""Image-level, stratified train/val/test split for TACO.

Two properties matter here and are easy to get wrong:

1. **No leakage.** TACO annotations are per-object, and one image can hold
   several objects. Splitting at the *annotation* level would let a model
   see one plastic bottle from image X during training and be evaluated on
   a different object cropped from that same image X during validation —
   the model has effectively already seen that background/lighting/scene.
   We split at the *image* level: every annotation belonging to an image
   goes into the same split as that image, no exceptions.

2. **Stratified by dominant material.** TACO's classes are heavily
   imbalanced. A naive random split can (and empirically does, on a
   dataset this small) starve a rare class out of the validation set by
   chance, making its eval metric meaningless. We stratify by each image's
   *dominant* material group (majority vote across its own annotations,
   ties broken by a fixed class-id order for determinism) so rare classes
   are spread proportionally across splits rather than concentrated in one.

No sklearn dependency — this is a small, fully-deterministic manual
implementation so the only source of randomness is the explicit `seed`.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.taxonomy import MATERIAL_GROUPS, get_material_group

# Deterministic tie-break order when an image has an equal count of two
# material groups — always prefer the earlier group in MATERIAL_GROUPS.
_GROUP_RANK = {name: i for i, name in enumerate(MATERIAL_GROUPS)}


def dominant_material_group(
    image_id: int, categories_by_ann: list[str]
) -> str | None:
    """Majority-vote material group for one image's list of category names.

    Returns None for images with zero annotations (they still get a split
    assignment, just not a stratification key — handled by the caller).
    """
    if not categories_by_ann:
        return None
    groups = [get_material_group(c).value for c in categories_by_ann]
    counts = Counter(groups)
    max_count = max(counts.values())
    tied = [g for g, c in counts.items() if c == max_count]
    return min(tied, key=lambda g: _GROUP_RANK[g])


def stratified_split(
    image_ids: list[int],
    stratum_by_image: dict[int, str | None],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Split image ids into train/val/test, stratified by `stratum_by_image`.

    Images with a None stratum (no annotations) are pooled into their own
    "unlabeled" bucket and split with the same ratios, rather than dropped
    — an image with no litter is still valid background/negative signal
    for a detector and shouldn't silently disappear from the dataset.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    by_stratum: dict[str, list[int]] = defaultdict(list)
    for img_id in image_ids:
        key = stratum_by_image.get(img_id) or "__unlabeled__"
        by_stratum[key].append(img_id)

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for _stratum, ids in by_stratum.items():
        ids = sorted(ids)  # deterministic order before shuffling
        rng.shuffle(ids)
        n = len(ids)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        # test gets the remainder so rounding never drops or duplicates an id
        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train : n_train + n_val])
        splits["test"].extend(ids[n_train + n_val :])

    return splits


def verify_no_leakage(splits: dict[str, list[int]]) -> None:
    """Raise AssertionError if any image id appears in more than one split."""
    seen: dict[int, str] = {}
    for split_name, ids in splits.items():
        for img_id in ids:
            if img_id in seen:
                raise AssertionError(
                    f"Leakage detected: image {img_id} is in both "
                    f"'{seen[img_id]}' and '{split_name}'"
                )
            seen[img_id] = split_name


def split_from_coco(
    annotations_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """End-to-end: read TACO's annotations.json, return per-split image records.

    Each returned record is the original COCO `images` entry (id, file_name,
    width, height, ...) so callers have everything needed to locate the
    actual image file without re-parsing the json.
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images_by_id = {img["id"]: img for img in coco["images"]}

    cats_by_image: dict[int, list[str]] = defaultdict(list)
    for ann in coco["annotations"]:
        cats_by_image[ann["image_id"]].append(categories[ann["category_id"]])

    strata = {
        img_id: dominant_material_group(img_id, cats_by_image.get(img_id, []))
        for img_id in images_by_id
    }

    id_splits = stratified_split(
        list(images_by_id.keys()), strata, train_ratio, val_ratio, test_ratio, seed
    )
    verify_no_leakage(id_splits)

    return {
        split_name: [images_by_id[i] for i in ids]
        for split_name, ids in id_splits.items()
    }


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = split_from_coco(args.annotations, seed=args.seed)
    for name, records in result.items():
        logging.info("%s: %d images", name, len(records))
