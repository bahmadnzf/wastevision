from __future__ import annotations

import pytest

from src.data.split import (
    dominant_material_group,
    split_from_coco,
    stratified_split,
    verify_no_leakage,
)


def test_dominant_material_group_majority_vote():
    # 2x plastic categories, 1x glass -> plastic wins.
    result = dominant_material_group(
        1, ["Clear plastic bottle", "Other plastic bottle", "Glass bottle"]
    )
    assert result == "plastic"


def test_dominant_material_group_empty_returns_none():
    assert dominant_material_group(1, []) is None


def test_dominant_material_group_tie_break_is_deterministic():
    # One glass, one metal -> tie. Must resolve the same way every call.
    r1 = dominant_material_group(1, ["Glass bottle", "Aluminium foil"])
    r2 = dominant_material_group(1, ["Aluminium foil", "Glass bottle"])
    assert r1 == r2


def test_stratified_split_covers_every_id_exactly_once():
    ids = list(range(50))
    strata = {i: ("plastic" if i % 2 == 0 else "metal") for i in ids}
    splits = stratified_split(ids, strata, seed=1)

    all_assigned = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_assigned) == ids
    verify_no_leakage(splits)  # should not raise


def test_stratified_split_approximate_ratios():
    ids = list(range(1000))
    strata = {i: "plastic" for i in ids}
    splits = stratified_split(ids, strata, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1)

    assert 750 <= len(splits["train"]) <= 850
    assert 50 <= len(splits["val"]) <= 150
    assert 50 <= len(splits["test"]) <= 150


def test_stratified_split_rejects_bad_ratios():
    with pytest.raises(ValueError):
        stratified_split([1, 2, 3], {1: "a", 2: "a", 3: "a"}, train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


def test_stratified_split_handles_unlabeled_images():
    ids = [1, 2, 3]
    strata = {1: "plastic", 2: None, 3: None}
    splits = stratified_split(ids, strata, seed=1)
    all_assigned = splits["train"] + splits["val"] + splits["test"]
    assert sorted(all_assigned) == ids


def test_verify_no_leakage_raises_on_overlap():
    bad_splits = {"train": [1, 2, 3], "val": [3, 4], "test": [5]}
    with pytest.raises(AssertionError):
        verify_no_leakage(bad_splits)


def test_split_from_coco_no_leakage_and_full_coverage(synthetic_coco_path):
    result = split_from_coco(synthetic_coco_path, seed=7)
    all_ids = [rec["id"] for records in result.values() for rec in records]

    assert len(all_ids) == len(set(all_ids))  # no image counted twice

    import json

    coco = json.loads(synthetic_coco_path.read_text())
    assert sorted(all_ids) == sorted(img["id"] for img in coco["images"])


def test_split_from_coco_is_deterministic_for_fixed_seed(synthetic_coco_path):
    result_a = split_from_coco(synthetic_coco_path, seed=123)
    result_b = split_from_coco(synthetic_coco_path, seed=123)
    for split_name in ("train", "val", "test"):
        ids_a = [r["id"] for r in result_a[split_name]]
        ids_b = [r["id"] for r in result_b[split_name]]
        assert ids_a == ids_b


def test_split_from_coco_different_seeds_can_differ(synthetic_coco_path):
    result_a = split_from_coco(synthetic_coco_path, seed=1)
    result_b = split_from_coco(synthetic_coco_path, seed=2)
    ids_a = [r["id"] for r in result_a["train"]]
    ids_b = [r["id"] for r in result_b["train"]]
    assert ids_a != ids_b
