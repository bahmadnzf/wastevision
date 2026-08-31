"""Evaluate a trained checkpoint and write a per-class report.

Usage:
    python -m src.models.evaluate --weights models/best.pt --data data/processed/data.yaml

Writes eval_results/report.md (overall + per-class mAP50 / mAP50-95) and
prints the same table to stdout — the report.md numbers are what should be
copy-pasted into CASE_STUDY.md section 7, and the per-class breakdown is the
starting point for the error-analysis pass in section 8 (start by pulling
validation examples from whichever class has the lowest AP50 here).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils.taxonomy import MATERIAL_GROUPS

logger = logging.getLogger(__name__)


def select_operating_threshold(px, f1_curve) -> tuple[float, float]:
    """Pick the confidence threshold that maximizes mean F1 across classes.

    Takes the raw arrays ultralytics exposes as `metrics.box.px`
    (1000 threshold steps from 0 to 1) and `metrics.box.f1_curve`
    (per-class F1 at each of those steps, shape [n_classes_present, 1000]
    or already 1-D if only one class is present). Deliberately takes plain
    arrays rather than the ultralytics metrics object so this is testable
    with synthetic data — no model, no val split, no ultralytics import
    needed to verify the argmax-selection logic itself.

    The API's default inference confidence (0.25 in `configs/config.yaml`
    and `src/models/inference.py`) is a reasonable generic starting point,
    not a tuned one — this is what tunes it against a specific trained
    checkpoint's actual precision/recall trade-off.
    """
    import numpy as np

    px_arr = np.asarray(px, dtype=float)
    f1_arr = np.asarray(f1_curve, dtype=float)
    mean_f1 = f1_arr.mean(axis=0) if f1_arr.ndim == 2 else f1_arr
    best_idx = int(np.argmax(mean_f1))
    return float(px_arr[best_idx]), float(mean_f1[best_idx])


def select_operating_threshold_from_metrics(metrics) -> tuple[float, float]:
    """Convenience wrapper for a real ultralytics DetMetrics object (as
    returned by `YOLO(...).val()`) — thin, not unit tested itself since it
    needs a real val run; the selection logic it delegates to is."""
    return select_operating_threshold(metrics.box.px, metrics.box.f1_curve)


def evaluate(weights_path: Path, data_yaml: Path, split: str = "test", imgsz: int = 640) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), split=split, imgsz=imgsz)

    overall = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    # IMPORTANT: metrics.box.ap50 / metrics.box.ap are NOT indexed by class
    # id — they're only as long as the number of classes that actually had
    # ground-truth instances in this split, and `ap_class_index` gives the
    # real class id each entry corresponds to. A class with zero instances
    # in the split (common for rare groups on a small dataset) is simply
    # absent from these arrays, not present with a zero. Indexing by
    # position instead of `ap_class_index` silently attributes one class's
    # AP to a different class's name — this is worth calling out because
    # it's an easy, quiet-failure mistake to make against this API.
    per_class = {name: {"AP50": None, "AP50-95": None} for name in MATERIAL_GROUPS}
    for position, class_id in enumerate(metrics.box.ap_class_index):
        name = MATERIAL_GROUPS[int(class_id)]
        per_class[name] = {
            "AP50": float(metrics.box.ap50[position]),
            "AP50-95": float(metrics.box.ap[position]),
        }

    return {"overall": overall, "per_class": per_class}


def write_report(results: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# WasteVision evaluation report", ""]
    lines.append("## Overall")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k, v in results["overall"].items():
        lines.append(f"| {k} | {v:.4f} |")

    lines.append("")
    lines.append("## Per-class")
    lines.append("| Material group | AP50 | AP50-95 |")
    lines.append("|---|---|---|")
    ranked = sorted(
        results["per_class"].items(),
        key=lambda kv: (kv[1]["AP50"] is None, kv[1]["AP50"] or 0.0),
    )
    for name, m in ranked:
        ap50 = f"{m['AP50']:.4f}" if m["AP50"] is not None else "n/a"
        ap = f"{m['AP50-95']:.4f}" if m["AP50-95"] is not None else "n/a"
        lines.append(f"| {name} | {ap50} | {ap} |")
    lines.append("")
    lines.append(
        "Rows are sorted worst-to-best by AP50 — the top row is the "
        "starting point for CASE_STUDY.md section 8's error analysis."
    )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-dir", type=Path, default=Path("eval_results"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    res = evaluate(args.weights, args.data, args.split, args.imgsz)
    path = write_report(res, args.out_dir)
    logger.info("Wrote %s", path)
    print(f"mAP50: {res['overall']['mAP50']:.4f}   mAP50-95: {res['overall']['mAP50-95']:.4f}")
