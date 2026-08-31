"""Minority-class oversampling for the prepared YOLO train split.

TACO's 8 collapsed material groups are still imbalanced after the 60->8
collapse (see CASE_STUDY.md section 3) — on the full dataset, "other"
(dominated by cigarette butts) and "plastic" vastly outnumber groups like
"organic" or "textile". Left alone, a detector trained on this split will
under-learn the minority groups.

This module oversamples by duplicating whole *images* (not individual
boxes) whose dominant material group is under-represented, up to some
fraction of the majority group's count. Duplicating at the image level
keeps every duplicated object's full scene context intact — it's the
same "don't split objects from the same image across examples" principle
`src/data/split.py` applies to avoid leakage, applied here to avoid
inventing synthetic-looking single-box training examples instead.

This is a real, if blunt, lever. It does not fix genuine within-class
visual diversity gaps — a class oversampled from 5 unique images to 40
copies still only has 5 unique images' worth of visual variety. Treat it
as a starting point, not a substitute for more labeled data.
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from src.utils.taxonomy import MATERIAL_GROUPS

_GROUP_RANK = {name: i for i, name in enumerate(MATERIAL_GROUPS)}


def dominant_group_from_label_file(label_path: Path) -> str | None:
    """Majority-vote material group for one YOLO label file's class ids.

    Mirrors `src.data.split.dominant_material_group`'s tie-break rule
    (prefer the earlier group in MATERIAL_GROUPS) but reads already-
    collapsed YOLO class ids from a label file, not raw TACO category
    names — this runs *after* data prep, on the output of
    `download_and_prepare.py`, not on the original COCO json.
    """
    if not label_path.exists():
        return None
    lines = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    class_ids = [int(ln.split()[0]) for ln in lines]
    groups = [MATERIAL_GROUPS[c] for c in class_ids]
    counts = Counter(groups)
    max_count = max(counts.values())
    tied = [g for g, c in counts.items() if c == max_count]
    return min(tied, key=lambda g: _GROUP_RANK[g])


def rebalance_train_split(
    processed_dir: Path, min_fraction_of_majority: float = 0.5
) -> dict:
    """Oversample train images so every material group reaches at least
    `min_fraction_of_majority` of the largest group's image count.

    Only touches `<processed_dir>/train/{images,labels}` — val and test
    are left untouched on purpose, so evaluation metrics stay honest and
    aren't inflated by evaluating on duplicated examples.

    Idempotent: re-running with the same `min_fraction_of_majority` on an
    already-rebalanced split adds nothing new (duplicate filenames are
    skipped), so this is safe to call again after changing the ratio.
    """
    images_dir = processed_dir / "train" / "images"
    labels_dir = processed_dir / "train" / "labels"

    by_group: dict[str, list[Path]] = {}
    for img_path in sorted(images_dir.iterdir()):
        group = dominant_group_from_label_file(labels_dir / f"{img_path.stem}.txt")
        if group is None:
            continue  # background/negative images: left alone, not duplicated
        by_group.setdefault(group, []).append(img_path)

    if not by_group:
        return {"added": 0, "before": {}, "after": {}}

    before_counts = {g: len(v) for g, v in by_group.items()}
    target = max(1, int(max(before_counts.values()) * min_fraction_of_majority))

    added = 0
    after_counts = dict(before_counts)
    for group, images in by_group.items():
        n = len(images)
        if n == 0 or n >= target:
            continue
        needed = target - n
        for i in range(needed):
            src_img = images[i % n]
            src_lbl = labels_dir / f"{src_img.stem}.txt"
            dup_stem = f"{src_img.stem}__dup{i}"
            dup_img = images_dir / f"{dup_stem}{src_img.suffix}"
            if dup_img.exists():
                continue
            shutil.copy2(src_img, dup_img)
            shutil.copy2(src_lbl, labels_dir / f"{dup_stem}.txt")
            added += 1
        after_counts[group] = n + needed

    return {"added": added, "before": before_counts, "after": after_counts}


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--min-fraction-of-majority", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = rebalance_train_split(args.processed_dir, args.min_fraction_of_majority)
    logging.info("Rebalance result: %s", result)
