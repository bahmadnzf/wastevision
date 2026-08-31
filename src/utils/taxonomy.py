"""
Taxonomy and recyclability policy for WasteVision.

This module is the single place where two *separate* concerns live side by
side on purpose:

1. PERCEPTION taxonomy — how TACO's 60 raw annotation categories collapse
   into the 8 material groups the detector is actually trained to predict.
2. POLICY taxonomy — how a predicted material group maps to a recyclability
   verdict. This is NOT a model artifact. It is a plain Python dict that a
   deployer edits for their own municipality's rules, with no retraining.

Why the collapse happens at all: several of TACO's 60 categories have
single-digit instance counts across the whole (~1500 image) dataset, which
is not enough signal for a detector to learn a reliable, distinct class
boundary. Collapsing to 8 material groups keeps every class statistically
learnable while still being the axis that actually drives a recyclability
decision (nobody sorts a facility line by "carded blister pack" vs. "other
plastic container" — they sort by material).

A known, explicitly documented limitation: because the trained detector
only ever outputs one of the 8 MATERIAL_GROUPS, it cannot distinguish, at
inference time, a food can from an aerosol can, or a battery from a strip
of scrap metal — they've all been collapsed into "metal". Where that
distinction is safety-relevant (batteries, aerosols, cigarette filters),
this module does not pretend the model can recover it. Instead the *policy
layer* marks the whole material group as `special_handling=True` with a
note explaining why, so a human (or a downstream rule) treats the group
with appropriate caution rather than assuming the model silently solved a
problem it structurally cannot solve. That's a more honest design than
threading a fake "hazard" prediction through a model that was never trained
to make it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MaterialGroup(str, Enum):
    PLASTIC = "plastic"
    METAL = "metal"
    GLASS = "glass"
    PAPER_CARDBOARD = "paper_cardboard"
    ORGANIC = "organic"
    TEXTILE = "textile"
    COMPOSITE = "composite"
    OTHER = "other"


# Fixed, ordered class list. Index position IS the YOLO class id — order must
# never change once training data has been generated against it, or every
# existing label file silently points at the wrong class.
MATERIAL_GROUPS: list[str] = [g.value for g in MaterialGroup]


class RecyclabilityVerdict(str, Enum):
    RECYCLABLE = "recyclable"
    NOT_RECYCLABLE = "not_recyclable"
    CHECK_LOCAL_PROGRAM = "check_local_program"


# ---------------------------------------------------------------------------
# 1. PERCEPTION: TACO's 60 categories -> 8 material groups
# ---------------------------------------------------------------------------
# Source category names match TACO's official annotations.json `categories`
# list (Proença & Simões, 2020, https://github.com/pedropro/TACO).
TACO_TO_MATERIAL_GROUP: dict[str, MaterialGroup] = {
    "Aluminium foil": MaterialGroup.METAL,
    "Battery": MaterialGroup.METAL,  # hazardous — see special_handling note on METAL
    "Aluminium blister pack": MaterialGroup.COMPOSITE,  # foil/plastic laminate
    "Carded blister pack": MaterialGroup.COMPOSITE,  # cardboard/plastic laminate
    "Other plastic bottle": MaterialGroup.PLASTIC,
    "Clear plastic bottle": MaterialGroup.PLASTIC,
    "Glass bottle": MaterialGroup.GLASS,
    "Plastic bottle cap": MaterialGroup.PLASTIC,
    "Metal bottle cap": MaterialGroup.METAL,
    "Broken glass": MaterialGroup.GLASS,  # handling hazard — see special_handling note on GLASS
    "Food Can": MaterialGroup.METAL,
    "Aerosol": MaterialGroup.METAL,  # hazardous (pressurized) — see special_handling note on METAL
    "Drink can": MaterialGroup.METAL,
    "Toilet tube": MaterialGroup.PAPER_CARDBOARD,
    "Other carton": MaterialGroup.PAPER_CARDBOARD,
    "Egg carton": MaterialGroup.PAPER_CARDBOARD,
    "Drink carton": MaterialGroup.COMPOSITE,  # e.g. Tetra Pak: paper/plastic/foil
    "Corrugated carton": MaterialGroup.PAPER_CARDBOARD,
    "Meal carton": MaterialGroup.COMPOSITE,  # typically coated/laminated
    "Pizza box": MaterialGroup.PAPER_CARDBOARD,
    "Paper cup": MaterialGroup.COMPOSITE,  # plastic/wax lining, not plain paper
    "Disposable plastic cup": MaterialGroup.PLASTIC,
    "Foam cup": MaterialGroup.PLASTIC,  # polystyrene
    "Glass cup": MaterialGroup.GLASS,
    "Other plastic cup": MaterialGroup.PLASTIC,
    "Food waste": MaterialGroup.ORGANIC,
    "Glass jar": MaterialGroup.GLASS,
    "Plastic lid": MaterialGroup.PLASTIC,
    "Metal lid": MaterialGroup.METAL,
    "Other plastic": MaterialGroup.PLASTIC,
    "Magazine paper": MaterialGroup.PAPER_CARDBOARD,
    "Tissues": MaterialGroup.PAPER_CARDBOARD,
    "Wrapping paper": MaterialGroup.PAPER_CARDBOARD,
    "Normal paper": MaterialGroup.PAPER_CARDBOARD,
    "Paper bag": MaterialGroup.PAPER_CARDBOARD,
    "Plastified paper bag": MaterialGroup.COMPOSITE,
    "Plastic film": MaterialGroup.PLASTIC,
    "Six pack rings": MaterialGroup.PLASTIC,
    "Garbage bag": MaterialGroup.PLASTIC,
    "Other plastic wrapper": MaterialGroup.PLASTIC,
    "Single-use carrier bag": MaterialGroup.PLASTIC,
    "Polypropylene bag": MaterialGroup.PLASTIC,
    "Crisp packet": MaterialGroup.COMPOSITE,  # metallized film laminate
    "Spread tub": MaterialGroup.PLASTIC,
    "Tupperware": MaterialGroup.PLASTIC,
    "Disposable food container": MaterialGroup.PLASTIC,
    "Foam food container": MaterialGroup.PLASTIC,
    "Other plastic container": MaterialGroup.PLASTIC,
    "Plastic glooves": MaterialGroup.PLASTIC,  # sic — matches TACO's own spelling
    "Plastic utensils": MaterialGroup.PLASTIC,
    "Pop tab": MaterialGroup.METAL,
    "Rope & strings": MaterialGroup.TEXTILE,
    "Scrap metal": MaterialGroup.METAL,
    "Shoe": MaterialGroup.COMPOSITE,  # rubber/textile/foam mix
    "Squeezable tube": MaterialGroup.COMPOSITE,  # plastic/metal laminate
    "Plastic straw": MaterialGroup.PLASTIC,
    "Paper straw": MaterialGroup.PAPER_CARDBOARD,
    "Styrofoam piece": MaterialGroup.PLASTIC,
    "Unlabeled litter": MaterialGroup.OTHER,
    "Cigarette": MaterialGroup.OTHER,  # cellulose acetate filter, toxic — see OTHER note
}


def get_material_group(taco_category: str) -> MaterialGroup:
    """Map a raw TACO category name to its collapsed material group.

    Raises KeyError with the offending name (not a generic message) so a
    typo or a TACO dataset update that renames a category fails loudly
    during data prep rather than silently mis-labeling training data.
    """
    try:
        return TACO_TO_MATERIAL_GROUP[taco_category]
    except KeyError as exc:
        raise KeyError(
            f"Unknown TACO category {taco_category!r} — not present in "
            "TACO_TO_MATERIAL_GROUP. If TACO has added/renamed a category, "
            "update the mapping in src/utils/taxonomy.py before re-running "
            "data prep."
        ) from exc


# ---------------------------------------------------------------------------
# 2. POLICY: material group -> recyclability verdict
# ---------------------------------------------------------------------------
# This is a *generic baseline*, not any specific municipality's real rules.
# It exists to be edited, not trusted as-is. Swap it per deployment.


@dataclass(frozen=True)
class Policy:
    verdict: RecyclabilityVerdict
    special_handling: bool
    note: str


RECYCLABILITY_POLICY: dict[MaterialGroup, Policy] = {
    MaterialGroup.PLASTIC: Policy(
        verdict=RecyclabilityVerdict.CHECK_LOCAL_PROGRAM,
        special_handling=False,
        note=(
            "Recyclability depends on resin type, which this taxonomy does "
            "not distinguish. Most rigid containers/bottles are accepted "
            "curbside in many programs; films, foams, and utensils often "
            "are not. Confirm against the local program's accepted-items "
            "list."
        ),
    ),
    MaterialGroup.METAL: Policy(
        verdict=RecyclabilityVerdict.RECYCLABLE,
        special_handling=True,
        note=(
            "Most metal (cans, foil, scrap) is broadly recyclable. However "
            "this group also absorbs batteries and aerosol cans, which the "
            "detector cannot distinguish from ordinary cans at this "
            "granularity. Route to hazardous/e-waste collection if a human "
            "check confirms either — do not compact or bin with clean "
            "metal recycling."
        ),
    ),
    MaterialGroup.GLASS: Policy(
        verdict=RecyclabilityVerdict.RECYCLABLE,
        special_handling=True,
        note=(
            "Glass is broadly recyclable, but this group includes broken "
            "glass, which is a laceration hazard for sorters/handlers even "
            "though the material itself is recyclable. Flag for careful "
            "handling regardless of the recyclability verdict."
        ),
    ),
    MaterialGroup.PAPER_CARDBOARD: Policy(
        verdict=RecyclabilityVerdict.RECYCLABLE,
        special_handling=False,
        note=(
            "Clean, dry paper/cardboard is broadly recyclable. Food- or "
            "liquid-contaminated items (greasy pizza box, used tissue) "
            "should go to compost/landfill instead — this taxonomy cannot "
            "detect contamination, only material, so treat this verdict as "
            "conditional on visible cleanliness."
        ),
    ),
    MaterialGroup.ORGANIC: Policy(
        verdict=RecyclabilityVerdict.NOT_RECYCLABLE,
        special_handling=False,
        note="Route to composting/organics stream, not recycling.",
    ),
    MaterialGroup.TEXTILE: Policy(
        verdict=RecyclabilityVerdict.NOT_RECYCLABLE,
        special_handling=False,
        note=(
            "Not accepted by standard curbside recycling. Route to textile "
            "take-back or donation programs where available."
        ),
    ),
    MaterialGroup.COMPOSITE: Policy(
        verdict=RecyclabilityVerdict.NOT_RECYCLABLE,
        special_handling=False,
        note=(
            "Multi-material laminates (crisp packets, drink cartons, paper "
            "cups) are generally rejected by standard single-stream "
            "recycling because the layers can't be separated economically. "
            "Check for a specialized program (e.g. carton-specific "
            "recycling, TerraCycle-style take-back) before defaulting to "
            "landfill."
        ),
    ),
    MaterialGroup.OTHER: Policy(
        verdict=RecyclabilityVerdict.NOT_RECYCLABLE,
        special_handling=True,
        note=(
            "Unidentifiable litter or inherently hazardous/toxic items "
            "(e.g. cigarette filters leach chemicals and are not "
            "recyclable). Route to general waste; treat as needing a "
            "manual check before assuming safe handling."
        ),
    ),
}


def get_verdict(material_group: MaterialGroup | str) -> Policy:
    """Look up the recyclability policy for a material group.

    Accepts either a MaterialGroup or its string value so API/inference
    code that has already round-tripped through JSON doesn't need to
    re-wrap in the enum.
    """
    group = MaterialGroup(material_group)
    return RECYCLABILITY_POLICY[group]
