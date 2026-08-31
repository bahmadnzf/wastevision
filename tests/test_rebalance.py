from __future__ import annotations

from pathlib import Path

import pytest

from src.data.rebalance import dominant_group_from_label_file, rebalance_train_split


def _write_label(path: Path, class_ids: list[int]) -> None:
    path.write_text("\n".join(f"{c} 0.5 0.5 0.1 0.1" for c in class_ids) + "\n" if class_ids else "")


def test_dominant_group_from_label_file_majority(tmp_path: Path):
    label = tmp_path / "img1.txt"
    _write_label(label, [0, 0, 1])  # plastic, plastic, metal -> plastic wins
    assert dominant_group_from_label_file(label) == "plastic"


def test_dominant_group_from_label_file_empty_returns_none(tmp_path: Path):
    label = tmp_path / "img1.txt"
    _write_label(label, [])
    assert dominant_group_from_label_file(label) is None


def test_dominant_group_from_label_file_missing_file_returns_none(tmp_path: Path):
    assert dominant_group_from_label_file(tmp_path / "does_not_exist.txt") is None


def _build_train_split(tmp_path: Path, group_counts: dict[str, int]) -> Path:
    """Build a minimal train/{images,labels} dir with `n` images per group,
    each image's single label matching that group's class id."""
    from src.utils.taxonomy import MATERIAL_GROUPS

    processed = tmp_path / "processed"
    images_dir = processed / "train" / "images"
    labels_dir = processed / "train" / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    for group, n in group_counts.items():
        class_id = MATERIAL_GROUPS.index(group)
        for i in range(n):
            stem = f"{group}_{i}"
            (images_dir / f"{stem}.jpg").write_bytes(b"fake-jpg-bytes")
            _write_label(labels_dir / f"{stem}.txt", [class_id])

    return processed


def test_rebalance_brings_minority_groups_up_to_target_fraction(tmp_path: Path):
    processed = _build_train_split(tmp_path, {"plastic": 20, "organic": 2, "textile": 4})
    result = rebalance_train_split(processed, min_fraction_of_majority=0.5)

    # target = 0.5 * 20 = 10
    assert result["after"]["plastic"] == 20  # majority untouched
    assert result["after"]["organic"] == 10
    assert result["after"]["textile"] == 10
    assert result["added"] == (10 - 2) + (10 - 4)


def test_rebalance_leaves_val_and_test_untouched(tmp_path: Path):
    processed = _build_train_split(tmp_path, {"plastic": 10, "organic": 1})
    # Create a val split with its own (untouched-by-design) images.
    val_images = processed / "val" / "images"
    val_labels = processed / "val" / "labels"
    val_images.mkdir(parents=True)
    val_labels.mkdir(parents=True)
    (val_images / "v1.jpg").write_bytes(b"x")
    _write_label(val_labels / "v1.txt", [0])

    rebalance_train_split(processed, min_fraction_of_majority=0.5)

    assert len(list(val_images.iterdir())) == 1  # unchanged


def test_rebalance_every_image_still_has_a_matching_label(tmp_path: Path):
    processed = _build_train_split(tmp_path, {"plastic": 15, "glass": 1, "metal": 3})
    rebalance_train_split(processed, min_fraction_of_majority=0.6)

    images_dir = processed / "train" / "images"
    labels_dir = processed / "train" / "labels"
    for img in images_dir.iterdir():
        assert (labels_dir / f"{img.stem}.txt").exists()


def test_rebalance_is_idempotent(tmp_path: Path):
    processed = _build_train_split(tmp_path, {"plastic": 10, "organic": 1})
    first = rebalance_train_split(processed, min_fraction_of_majority=0.5)
    second = rebalance_train_split(processed, min_fraction_of_majority=0.5)
    assert second["added"] == 0
    assert first["after"] == second["before"]


def test_rebalance_empty_train_split_returns_zero(tmp_path: Path):
    processed = tmp_path / "processed"
    (processed / "train" / "images").mkdir(parents=True)
    (processed / "train" / "labels").mkdir(parents=True)
    result = rebalance_train_split(processed)
    assert result == {"added": 0, "before": {}, "after": {}}
