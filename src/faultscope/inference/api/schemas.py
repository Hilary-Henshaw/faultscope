"""Pydantic v2 request and response schemas for the inference API.

All request models validate inputs at the FastAPI layer before they
reach the ``PredictionEngine``.  All response models define the exact
JSON contract returned to callers.

Design notes
------------
- ``Field(...)`` marks required fields; defaults make fields optional.
- ``min_length`` / ``max_length`` constraints on ``str`` and ``list``
  are enforced by Pydantic at parse time, producing 422 errors before
  any inference code runs.
- ``datetime`` fields are serialised as ISO-8601 UTC strings via the
  ``model_config`` setting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RulPredictionRequest(BaseModel):
    """Request body for ``POST /api/v1/predict/remaining-life``.

    Attributes
    ----------
    machine_id:
        Unique machine identifier.  1–64 characters.
    feature_sequence:
        Ordered list of feature dictionaries, oldest first.
        Between 1 and 200 timesteps.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    machine_id: str = Field(..., min_length=1, max_length=64)
    feature_sequence: list[dict[str, float]] = Field(
        ..., min_length=1, max_length=200
    )


class HealthPredictionRequest(BaseModel):
    """Request body for ``POST /api/v1/predict/health-status``.

    Attributes
    ----------
    machine_id:
        Unique machine identifier.  1–64 characters.
    features:
        Non-empty dictionary of feature name → numeric value.
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        populate_by_name=True,
    )

    machine_id: str = Field(..., min_length=1, max_length=64)
    features: dict[str, float] = Field(..., min_length=1)


class BatchPredictionItem(BaseModel):
    """A single item within a batch prediction request.

    Attributes
    ----------
    request_id:
        Caller-assigned correlation identifier returned verbatim in
        the response.
    prediction_type:
        ``"rul"`` for remaining-life regression, ``"health"`` for
        health classification.
    machine_id:
        Source machine identifier.
    feature_sequence:
        Required when ``prediction_type=="rul"``.
    features:
        Required when ``prediction_type=="health"``.
    """

    model_config = ConfigDict(populate_by_name=True)

    request_id: str = Field(..., min_length=1, max_length=128)
    prediction_type: Literal["rul", "health"]
    machine_id: str = Field(..., min_length=1, max_length=64)
    feature_sequence: list[dict[str, float]] | None = Field(
        default=None, max_length=200
    )
    features: dict[str, float] | None = None


class BatchPredictionRequest(BaseModel):
    """Request body for ``POST /api/v1/predict/batch``.

    Attributes
    ----------
    items:
        Between 1 and 100 ``BatchPredictionItem`` objects.
    """

    model_config = ConfigDict(populate_by_name=True)

    items: list[BatchPredictionItem] = Field(..., max_length=100)


class RulPredictionResponse(BaseModel):
    """Response body for ``POST /api/v1/predict/remaining-life``.

    Attributes
    ----------
    machine_id:
        Echo of the request machine_id.
    rul_cycles:
        Point estimate of remaining useful life in cycles.
    rul_hours:
        Point estimate in hours.
    rul_lower_bound:
        5th-percentile lower confidence bound (cycles).
    rul_upper_bound:
        95th-percentile upper confidence bound (cycles).
    health_label:
        Derived health category.
    confidence:
        Model confidence in ``[0, 1]``.
    model_version:
        MLflow Registry version string.
    predicted_at:
        UTC timestamp of the prediction.
    latency_ms:
        End-to-end inference latency in milliseconds.
    """

    model_config = ConfigDict(populate_by_name=True)

    machine_id: str
    rul_cycles: float
    rul_hours: float
    rul_lower_bound: float
    rul_upper_bound: float
    health_label: str
    confidence: float
    model_version: str
    predicted_at: datetime
    latency_ms: int


class HealthPredictionResponse(BaseModel):
    """Response body for ``POST /api/v1/predict/health-status``.

    Attributes
    ----------
    machine_id:
        Echo of the request machine_id.
    health_label:
        Predicted health category.
    probabilities:
        Per-class probability dictionary.
    model_version:
        MLflow Registry version string.
    predicted_at:
        UTC timestamp of the prediction.
    """

    model_config = ConfigDict(populate_by_name=True)

    machine_id: str
    health_label: str
    probabilities: dict[str, float]
    model_version: str
    predicted_at: datetime


class BatchPredictionResponse(BaseModel):
    """Response body for ``POST /api/v1/predict/batch``.

    Attributes
    ----------
    results:
        One result dict per input item, in the same order.
    batch_size:
        Number of items in the batch.
    total_latency_ms:
        Total wall-clock time from request receipt to response (ms).
    """

    model_config = ConfigDict(populate_by_name=True)

    results: list[dict[str, object]]
    batch_size: int
    total_latency_ms: int


class ModelCatalogResponse(BaseModel):
    """Response body for ``GET /api/v1/models``.

    Attributes
    ----------
    rul_model:
        Metadata dict for the active RUL model.
    health_model:
        Metadata dict for the active health model.
    last_reload:
        UTC timestamp of the most recent model load.
    """

    model_config = ConfigDict(populate_by_name=True)

    rul_model: dict[str, object]
    health_model: dict[str, object]
    last_reload: datetime


class HealthResponse(BaseModel):
    """Response body for ``GET /health``.

    Attributes
    ----------
    status:
        ``"ok"`` when the service is fully operational.
    models_loaded:
        Dict mapping model name → boolean loaded status.
    uptime_s:
        Service uptime in seconds.
    dependencies:
        Dict mapping dependency name → status string.
    """

    model_config = ConfigDict(populate_by_name=True)

    status: str
    models_loaded: dict[str, bool]
    uptime_s: float
    dependencies: dict[str, str]
