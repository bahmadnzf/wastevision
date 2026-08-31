# WasteVision

**Automated waste detection & recyclability classification from images.**

WasteVision detects individual waste items in a photo (street litter, a bin,
a materials-recovery-facility conveyor image), classifies each item's
material group, and applies a recyclability policy — the kind of pipeline
that could sit behind a smart-bin camera or a municipal waste audit tool.

Live demo: *(add your Hugging Face Spaces link here after deploying — see `DEPLOYMENT.md`)*

Full write-up of design decisions, results, and limitations: [`CASE_STUDY.md`](CASE_STUDY.md)

---

## Architecture

```
                    ┌─────────────────┐
   TACO dataset ──► │  Data pipeline   │  COCO → YOLO labels, stratified split
   (60 raw classes) │  (src/data/)     │  60 classes → 8 material groups
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Training         │  YOLOv8 transfer learning
                    │  (src/models/)    │  MLflow experiment tracking
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Inference        │  Detections → material →
                    │  wrapper          │  recyclability verdict
                    └───┬─────────┬─────┘
                        │         │
                ┌───────▼──┐  ┌───▼────────┐
                │ FastAPI  │  │ Gradio demo │
                │ REST API │  │ (HF Spaces) │
                └──────────┘  └─────────────┘
```

**Key design decision — perception vs. policy separation:** the model
predicts material groups; a separate, swappable mapping table converts
material → recyclability verdict. Recyclability rules vary by municipality,
so baking them into the trained weights would make the model wrong the
moment it's deployed somewhere else. See `src/utils/taxonomy.py` for the
full reasoning and the mapping table itself.

## Repo layout

```
src/
  data/           # COCO→YOLO conversion, stratified split, download/prep
  models/         # train.py, evaluate.py, inference.py
  api/            # FastAPI service
  utils/          # taxonomy mapping (material groups + recyclability policy)
app/
  gradio_app.py   # demo UI
tests/            # pytest suite (44 tests, run without GPU or real weights)
configs/
  config.yaml     # training hyperparameters
notebooks/
  train_colab.ipynb  # free-GPU training notebook
```

## Status

The full pipeline (taxonomy, data prep, training, evaluation, inference,
API, demo UI) is implemented and covered by tests, and has been run
end-to-end with the real `ultralytics`/`mlflow` stack against a small
synthetic dataset to catch integration bugs the unit tests can't (see
"Testing philosophy" below) — that run caught and fixed two real bugs:
an MLflow filesystem-store compatibility issue, and a per-class AP
mis-indexing bug in `evaluate.py` (ultralytics' `metrics.box.ap50` is
indexed by which classes appear in the eval split, not by class id —
`ap_class_index` is required to map correctly).

**No training has been run on the real TACO dataset yet** — that needs a
GPU and TACO's actual (Flickr-hosted) images, neither of which this
environment has. Section 7-8 of `CASE_STUDY.md` are still marked
`[FILL AFTER TRAINING]`: run `notebooks/train_colab.ipynb` (free GPU) or
`src/models/train.py` on your own hardware, then drop the real mAP numbers
and error-analysis findings in.

## Quickstart

### 1. Run the test suite (no GPU, no dataset needed)

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

All 44 tests pass using synthetic fixtures and a fake detector injected via
FastAPI's dependency-override system — the taxonomy mapping, the COCO→YOLO
converter, the leak-free stratified splitter, the inference enrichment
logic, and the API contract are all verified independently of having
trained weights.

### 2. Prepare the dataset

TACO's images are Flickr-hosted and fetched via the official downloader
(not redistributed directly). See `src/data/download_and_prepare.py` for
the full explanation and one-command pipeline once images are downloaded:

```bash
git clone https://github.com/pedropro/TACO.git && cd TACO && python download.py && cd ..
python -m src.data.download_and_prepare --images-dir TACO/data --output-dir data/processed
```

### 3. Train

Requires a GPU — use `notebooks/train_colab.ipynb` for free GPU access, or:

```bash
python -m src.models.train --config configs/config.yaml
```

Training runs are logged to MLflow (`mlflow ui` to browse experiments).
Local file-store tracking (the default) needs `MLFLOW_ALLOW_FILE_STORE=true`
on recent MLflow versions — `train.py` sets this automatically when
`mlflow_tracking_uri` in `config.yaml` is a local path rather than a
`sqlite://`/`http://` URI.

### 4. Evaluate

```bash
python -m src.models.evaluate --weights models/best.pt --data data/processed/data.yaml
```

Writes `eval_results/report.md` — overall + per-class mAP50/mAP50-95,
sorted worst-to-best, which is the starting point for the error-analysis
pass in `CASE_STUDY.md` section 8.

### 5. Serve

```bash
# API
docker build -t wastevision-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models:ro wastevision-api

# or both API + demo UI together
docker compose up
```

Then:
```bash
curl -X POST -F "file=@sample.jpg" http://localhost:8000/predict
```

See `DEPLOYMENT.md` for deploying the demo to Hugging Face Spaces.

## Testing philosophy

Every module that doesn't strictly require a GPU is unit tested against
synthetic data or a fake `Detector` (satisfying the same protocol
`YoloDetector` does, so it's a drop-in swap) — the taxonomy mapping, the
COCO→YOLO converter, the dataset splitter (including explicit leakage and
determinism checks), and the inference/API layers (via `InferenceService`
constructed with a fake detector, and FastAPI's dependency-override system
for the API tests). This means CI can verify correctness on every commit
without needing a GPU runner.

`requirements-dev.txt` deliberately does not pull in `ultralytics`,
`torch`, `mlflow`, or `gradio` — the test suite never imports them, by
design, so CI installs and runs fast.

## License

MIT — see `LICENSE`. TACO dataset is licensed separately by its authors;
see https://github.com/pedropro/TACO for terms.
