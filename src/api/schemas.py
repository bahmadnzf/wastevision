"""Pydantic request/response models for the WasteVision API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    material_group: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    recyclable: str
    special_handling: bool
    policy_note: str


class PredictResponse(BaseModel):
    detections: list[DetectionOut]
    count: int = Field(description="Number of detections above the confidence threshold")


class HealthResponse(BaseModel):
    status: str
