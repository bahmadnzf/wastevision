"""Train WasteVision's YOLOv8 detector, with every run logged to MLflow.

Usage:
    python -m src.models.train --config configs/config.yaml
    python -m src.models.train --config configs/config.yaml --model yolov8s.pt --epochs 150

CLI overrides take precedence over the YAML file so a sweep can be scripted
without editing config.yaml, e.g.:
    for m in yolov8n.pt yolov8s.pt yolov8m.pt; do
        python -m src.models.train --config configs/config.yaml --model $m
    done
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Keys in config.yaml that are WasteVision-specific bookkeeping, not
# ultralytics train() kwargs — stripped out before the ** unpack below.
_NON_ULTRALYTICS_KEYS = {"project", "experiment_name", "mlflow_tracking_uri"}


def load_config(config_path: Path, overrides: dict) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def train(config: dict) -> Path:
    """Runs training, returns the path to the best checkpoint."""
    import mlflow
    from ultralytics import YOLO
    from ultralytics import settings as ultralytics_settings

    # ultralytics ships its own built-in MLflow autologger (on by default),
    # which logs the run under an experiment named after `project` — a
    # different, overlapping run from the one this function logs explicitly
    # below. Left enabled, every training run creates two confusing,
    # partially-redundant MLflow entries. We do our own logging (with
    # WasteVision-specific params/metrics), so disable ultralytics' copy.
    ultralytics_settings.update({"mlflow": False})

    tracking_uri = config.get("mlflow_tracking_uri", "mlruns")
    # MLflow >=3 puts the plain local filesystem store into maintenance
    # mode and refuses to use it unless explicitly opted into, to steer
    # people toward a DB-backed store. A local file store is the right
    # default for a solo/portfolio project (zero extra services to run),
    # so opt in explicitly rather than forcing a DB dependency — swap
    # `mlflow_tracking_uri` in config.yaml to a `sqlite:///...` or `http://`
    # URI for team use instead.
    if not tracking_uri.startswith(("http://", "https://", "sqlite:///", "postgresql://", "mysql://")):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.get("experiment_name", "wastevision"))

    ultralytics_kwargs = {k: v for k, v in config.items() if k not in _NON_ULTRALYTICS_KEYS and k != "model"}

    with mlflow.start_run():
        mlflow.log_params({k: v for k, v in config.items() if k not in ("data",)})

        model = YOLO(config["model"])
        results = model.train(project=config.get("project", "runs/train"), **ultralytics_kwargs)

        # ultralytics writes its own metrics; pull the final epoch's
        # headline numbers into MLflow too so comparing runs doesn't
        # require cross-referencing two separate tools.
        try:
            metrics = results.results_dict
            mlflow.log_metrics(
                {
                    "mAP50": metrics.get("metrics/mAP50(B)", 0.0),
                    "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0.0),
                    "precision": metrics.get("metrics/precision(B)", 0.0),
                    "recall": metrics.get("metrics/recall(B)", 0.0),
                }
            )
        except Exception:  # pragma: no cover — best-effort metric logging
            logger.warning("Could not extract summary metrics for MLflow logging.", exc_info=True)

        best_weights = Path(model.trainer.best)
        mlflow.log_artifact(str(best_weights))
        logger.info("Best weights: %s", best_weights)
        return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--model", type=str, default=None, help="Override model in config, e.g. yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    overrides = {
        "model": args.model,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
    }
    cfg = load_config(args.config, overrides)
    train(cfg)
