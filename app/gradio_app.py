"""Gradio demo UI for WasteVision.

    python app/gradio_app.py
    # or: docker compose up (see DEPLOYMENT.md for HF Spaces)

Calls the exact same InferenceService used by the FastAPI service
(src/models/inference.py) instead of re-implementing detection, so the demo
and the production API share one code path — fixing a bug or swapping the
model happens once, not twice.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from src.models.inference import Detection, InferenceService, load_inference_service

WEIGHTS_PATH = os.environ.get("WASTEVISION_WEIGHTS", "/app/models/best.pt")

_service: InferenceService | None = None

_BOX_COLORS = {
    "plastic": "#3B82F6",
    "metal": "#9CA3AF",
    "glass": "#10B981",
    "paper_cardboard": "#D97706",
    "organic": "#65A30D",
    "textile": "#DB2777",
    "composite": "#7C3AED",
    "other": "#EF4444",
}


def get_service() -> InferenceService:
    global _service
    if _service is None:
        if not Path(WEIGHTS_PATH).exists():
            raise gr.Error(
                f"No weights found at {WEIGHTS_PATH}. Train a model first "
                "(`python -m src.models.train`) or set WASTEVISION_WEIGHTS "
                "to point at an existing checkpoint."
            )
        _service = load_inference_service(WEIGHTS_PATH)
    return _service


def _draw_boxes(image: Image.Image, detections: list[Detection]) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for det in detections:
        x0, y0, x1, y1 = det.bbox_xyxy
        color = _BOX_COLORS.get(det.material_group, "#EF4444")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = f"{det.material_group} {det.confidence:.0%}"
        label_h = 16
        draw.rectangle([x0, max(0, y0 - label_h), x0 + 8 * len(label), y0], fill=color)
        draw.text((x0 + 2, max(0, y0 - label_h)), label, fill="white")
    return annotated


def predict(image: Image.Image):
    if image is None:
        raise gr.Error("Upload an image first.")

    service = get_service()
    tmp_path = "/tmp/wastevision_input.jpg"
    image.convert("RGB").save(tmp_path)
    detections = service.predict(tmp_path)

    annotated = _draw_boxes(image, detections)
    table = [
        [
            d.material_group,
            f"{d.confidence:.0%}",
            d.recyclable,
            "yes" if d.special_handling else "no",
            d.policy_note,
        ]
        for d in detections
    ]
    return annotated, table


with gr.Blocks(title="WasteVision") as demo:
    gr.Markdown(
        "# WasteVision\n"
        "Upload a photo of waste items. The model detects each item, "
        "classifies its material group, and applies a baseline "
        "recyclability policy. **This policy is a generic default, not any "
        "specific municipality's actual rules** — see CASE_STUDY.md."
    )
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input image")
        output_image = gr.Image(type="pil", label="Detections")
    output_table = gr.Dataframe(
        headers=["Material", "Confidence", "Recyclable?", "Special handling?", "Note"],
        label="Detection detail",
    )
    submit = gr.Button("Detect", variant="primary")
    submit.click(fn=predict, inputs=input_image, outputs=[output_image, output_table])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
