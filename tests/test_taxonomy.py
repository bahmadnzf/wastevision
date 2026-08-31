from __future__ import annotations

import pytest

from src.utils.taxonomy import (
    MATERIAL_GROUPS,
    RECYCLABILITY_POLICY,
    TACO_TO_MATERIAL_GROUP,
    MaterialGroup,
    get_material_group,
    get_verdict,
)


def test_material_groups_has_eight_entries():
    assert len(MATERIAL_GROUPS) == 8
    assert len(set(MATERIAL_GROUPS)) == 8  # no duplicates


def test_material_groups_matches_enum_order():
    assert MATERIAL_GROUPS == [g.value for g in MaterialGroup]


def test_taco_mapping_has_sixty_categories():
    # TACO's published taxonomy: 60 leaf categories.
    assert len(TACO_TO_MATERIAL_GROUP) == 60


def test_every_taco_category_maps_to_a_valid_material_group():
    for category, group in TACO_TO_MATERIAL_GROUP.items():
        assert isinstance(group, MaterialGroup)
        assert group.value in MATERIAL_GROUPS


def test_get_material_group_known_category():
    assert get_material_group("Clear plastic bottle") == MaterialGroup.PLASTIC
    assert get_material_group("Glass bottle") == MaterialGroup.GLASS
    assert get_material_group("Food waste") == MaterialGroup.ORGANIC


def test_get_material_group_unknown_category_raises_with_name_in_message():
    with pytest.raises(KeyError) as exc_info:
        get_material_group("Definitely not a real TACO category")
    assert "Definitely not a real TACO category" in str(exc_info.value)


def test_hazardous_taco_categories_land_in_special_handling_groups():
    # Battery and Aerosol collapse into METAL; the METAL policy must flag
    # special_handling so a deployer doesn't treat "metal" as uniformly safe.
    assert get_material_group("Battery") == MaterialGroup.METAL
    assert get_material_group("Aerosol") == MaterialGroup.METAL
    assert RECYCLABILITY_POLICY[MaterialGroup.METAL].special_handling is True

    assert get_material_group("Cigarette") == MaterialGroup.OTHER
    assert RECYCLABILITY_POLICY[MaterialGroup.OTHER].special_handling is True


def test_every_material_group_has_a_policy():
    for group in MaterialGroup:
        assert group in RECYCLABILITY_POLICY
        policy = RECYCLABILITY_POLICY[group]
        assert policy.note  # non-empty rationale for every verdict


def test_get_verdict_accepts_enum_or_string():
    from_enum = get_verdict(MaterialGroup.GLASS)
    from_string = get_verdict("glass")
    assert from_enum == from_string


def test_get_verdict_unknown_string_raises():
    with pytest.raises(ValueError):
        get_verdict("not_a_material")
