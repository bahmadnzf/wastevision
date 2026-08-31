# WasteVision: Case Study

*This document is the narrative companion to the code — the part a reviewer
actually reads. Sections marked `[FILL AFTER TRAINING]` are placeholders:
run `notebooks/train_colab.ipynb` (or `src/models/train.py` on your own
GPU) on the full TACO dataset and drop your real numbers and
error-analysis findings in. Everything else reflects decisions already
made and verified in the codebase.*

## 0. Pipeline verification status

Every module in `src/` is implemented and covered by `tests/` (55 tests,
synthetic data + a fake detector, no GPU needed — see README's "Testing
philosophy"). Beyond the unit tests, the full path — data prep → train →
evaluate → inference → API/UI — has also been run end-to-end with the real
`ultralytics`/`mlflow` stack against a small synthetic dataset, specifically
to catch integration bugs unit tests with fakes structurally can't. That
run caught and fixed three real bugs, left here as a demonstration of why
this step matters, not swept under the rug:

- **MLflow filesystem-store compatibility.** Recent MLflow versions put
  the plain local file store into maintenance mode and refuse to use it
  without an explicit opt-in. `train.py` now sets this automatically for
  local tracking URIs.
- **Duplicate experiment logging.** `ultralytics` ships its own built-in
  MLflow autologger, on by default, which logged a second, differently-named,
  overlapping experiment alongside the one `train.py` logs explicitly.
  Now disabled in favor of the explicit logging, which captures
  WasteVision-specific params.
- **Per-class AP mis-indexing.** `metrics.box.ap50` (ultralytics'
  per-class average precision array) is only as long as the number of
  classes with ground-truth instances in that eval split, and is ordered
  by `metrics.box.ap_class_index`, not by class id. The first version of
  `evaluate.py` indexed it positionally, which — for any split missing at
  least one class, i.e. most small-dataset splits — would have silently
  attributed one class's AP to a different class's name instead of merely
  erroring. Fixed to map through `ap_class_index` explicitly.

**No training has been run on the real TACO dataset.** That needs a GPU
and TACO's actual (Flickr-hosted) images, neither available in the
environment this pipeline was built in. Sections 7-8 below are genuinely
unfilled, not abbreviated — that's the next step, via
`notebooks/train_colab.ipynb`.

## 1. Problem

Materials-recovery facilities and municipal waste-audit programs need to
know *what* is in a waste stream and *whether it's recyclable*, but manual
auditing doesn't scale and existing "smart bin" cameras mostly do single-item
classification rather than detecting multiple items in a cluttered scene.

WasteVision frames this as **object detection**, not classification: given
an image that may contain several waste items in realistic, cluttered
conditions, detect and localize each one, assign it a material group, and
apply a recyclability verdict.

## 2. Why detection, not classification

The obvious first instinct is an image classifier (à la TrashNet: one clean,
centered object per photo → one label). That doesn't match the deployment
scenario — real camera feeds see multiple overlapping items, partial
occlusion, and background clutter. Choosing TACO (real litter-in-context
photos with bounding-box annotations) over a cleaner classification dataset
was a deliberate trade: harder problem, noisier data, but a model that
actually matches how the system would be used.

## 3. The 60→8 class collapse

TACO's 60 annotation categories (verified against TACO's own
`categories.json` — see `src/utils/taxonomy.py`) are too fine-grained and
too imbalanced to train a reliable detector — several classes have
single-digit instance counts across the entire dataset.
`src/utils/taxonomy.py` documents the full mapping down to 8 material
groups (plastic, metal, glass, paper/cardboard, organic, textile,
composite, other), chosen because:

- each group has enough instances to actually learn from,
- the groups map onto decisions a recovery facility makes operationally,
- fine categories that matter for *safety* (batteries, aerosols) are
  handled honestly rather than optimistically: they collapse into the
  METAL group like any other metal item (the detector genuinely cannot
  distinguish a battery from a food can at this granularity), and the
  METAL entry in the recyclability policy is flagged
  `special_handling=True` with a note that the bucket may contain
  hazardous items — a caution baked into the policy layer, not a fake
  capability implied of the model.

If the eventual class distribution is still badly skewed after collapsing
to 8 groups (expected — TACO is cigarette/plastic-heavy), the training
pipeline includes `src/data/rebalance.py`, which oversamples whole images
(not individual boxes, to keep scene context intact) for under-represented
groups in the train split only, up to a configurable fraction of the
majority group's count.

## 4. Perception/policy separation

Recyclability isn't a fixed fact — it depends on the local municipal
program. Baking a specific recyclability ruleset into the trained model
weights would make the model *wrong by design* the moment someone deploys
it in a different city. Instead, the detector predicts material group only;
a separate, swappable mapping table (also in `taxonomy.py`) converts
material → verdict. Anyone deploying this elsewhere edits one Python dict,
not retrains a model. This is the single design decision worth defending
in an interview — it's a "modeling vs. product" distinction, not a
"technique" one.

## 5. Data split strategy

Split at the image level (not annotation level) to prevent leakage — an
image with multiple objects can't have some objects in train and others in
val. Stratified by each image's dominant material group so rare classes
aren't concentrated in one split by chance. See `src/data/split.py` and its
tests for the implementation and the leakage-prevention checks
(`verify_no_leakage`, exercised directly and via an end-to-end synthetic
COCO fixture in `tests/test_split.py`).

## 6. Model & training

- **Architecture:** YOLOv8 (nano → small → medium progression), transfer
  learning from COCO-pretrained weights.
- **Why YOLOv8:** real-time-capable (relevant if this ever runs on an
  edge camera at a facility), mature tooling, strong small-object
  performance which matters here (litter items are often small in frame).
- **Augmentation:** tuned for small, cluttered objects rather than
  YOLO's out-of-the-box defaults — upweighted box loss (`box: 8.0`),
  `close_mosaic` to disable mosaic augmentation for the final epochs
  (heavy mosaic can hurt small-object localization late in training), and
  moderate mixup given TACO's small size (~1,500 images) makes overfitting
  a real risk. See `configs/config.yaml` for the full rationale inline.
- **Class imbalance:** `src/data/rebalance.py` oversampling (section 3),
  applied before training whenever the class-distribution check
  (`notebooks/train_colab.ipynb` section 4) shows it's warranted.
- **Confidence threshold:** the API/inference default (0.25) is a generic
  starting point, not tuned. `src/models/evaluate.py::select_operating_threshold`
  picks the F1-maximizing threshold for a specific trained checkpoint from
  ultralytics' own F1-vs-confidence curve — done on the val split, with
  test held out for the final, unbiased report.
- **Experiment tracking:** every run logged to MLflow (hyperparameters,
  mAP50/mAP50-95, artifact weights) so runs are comparable, not just the
  final "best" one — including the architecture/hyperparameter sweep
  across yolov8n/s/m in the training notebook.

`[FILL AFTER TRAINING]`
- Final architecture chosen: ___
- Number of training runs compared: ___
- Key hyperparameter findings (e.g. augmentation strength, learning rate
  schedule, image size trade-offs): ___
- Tuned confidence threshold selected: ___ (vs. 0.25 generic default)
- Whether TTA evaluation was worth its inference-latency cost: ___

## 7. Results

`[FILL AFTER TRAINING]`

| Metric | Value |
|---|---|
| mAP50 (overall) | ___ |
| mAP50-95 (overall) | ___ |
| Per-class AP50 (worst class) | ___ — likely a rare/visually-ambiguous group; name it and explain why |
| Per-class AP50 (best class) | ___ |
| CPU inference latency (ONNX, ms/image) | ___ — see notebook section 12; informs whether the free HF Spaces CPU tier is viable |

## 8. Error analysis

`[FILL AFTER TRAINING]` — this section matters more than the metrics table
above for demonstrating engineering judgment. Pull 20-30 misclassified or
missed examples from the validation set and look for patterns:

- Which material groups get confused with each other, and why (visual
  similarity? annotation ambiguity in TACO itself — e.g. a soiled paper cup
  vs. a plastic cup at low resolution)?
- Does the model fail more on small objects, occluded objects, or specific
  lighting/background conditions?
- Are there labeling inconsistencies in TACO itself worth flagging?
- Did oversampling (section 3/6) actually help the minority groups it
  targeted, or did it just increase their false-positive rate by
  overfitting to a handful of duplicated images?

Write 3-5 concrete findings here with example images. This is what turns
"I trained a model" into "I understand what my model does and doesn't do."

## 9. Limitations

- TACO is a relatively small dataset (~1,500 images) for object detection;
  expect the model to generalize imperfectly to waste streams that look
  visually different (different countries' packaging, different lighting).
- The recyclability policy layer encodes a generic baseline, not any
  specific municipality's actual rules — explicitly not production-ready
  without local policy customization.
- Material-group granularity means visually similar materials (e.g. some
  composites vs. plastics) are an inherent source of error, not just a
  training artifact.
- Image-level oversampling (section 3) duplicates existing images rather
  than adding new visual variety — it corrects the loss function's
  attention to minority groups, not the underlying data scarcity for them.
- The METAL group's `special_handling` flag is a coarse, bucket-level
  caution (see section 3), not a per-item hazard detector — treat any
  METAL detection as "verify before assuming it's safe to recycle
  normally," not as a reliable battery/aerosol classifier.

## 10. What I'd do with more time/compute

- Active learning loop: use model uncertainty to prioritize which new
  images get labeled next, rather than random sampling.
- Semi-supervised pretraining on unlabeled waste imagery before fine-tuning
  on TACO's labeled subset, given how small TACO is.
- A lightweight on-device (quantized) version for actual edge deployment
  on a smart-bin camera, with a latency/accuracy trade-off analysis
  (`notebooks/train_colab.ipynb` section 12 does the ONNX/CPU half of this
  already; quantization and true edge-device benchmarking are the
  remaining step).
- Real photographic augmentation for the hazard-adjacent items in METAL
  (battery/aerosol) rather than relying solely on the policy-layer caution,
  if a per-item hazard signal ever becomes a real product requirement.
