"""Prediction endpoints for remaining-life and health-status inference.

Routes
------
POST /api/v1/predict/remaining-life
    Single-machine RUL prediction with MC-Dropout uncertainty bounds.

POST /api/v1/predict/health-status
    Single-machine health classification with class probabilities.

POST /api/v1/predict/batch
    Up to 100 predictions (any mix of RUL and health) in one call.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from faultscope.common.logging import get_logger
from faultscope.inference.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthPredictionRequest,
    HealthPredictionResponse,
    RulPredictionRequest,
    RulPredictionResponse,
)
from faultscope.inference.engine.predictor import (
    BatchPredictionItem as EngineBatchItem,
)
from faultscope.inference.engine.predictor import (
    PredictionEngine,
)

_log = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Predictions"])


@router.post(
    "/predict/remaining-life",
    response_model=RulPredictionResponse,
)
async def predict_remaining_life(
    request_body: RulPredictionRequest,
    request: Request,
) -> RulPredictionResponse:
    """Predict remaining useful life for a single machine.

    Accepts a time-ordered feature sequence and returns a RUL point
    estimate, confidence bounds, a health label, and inference
    metadata.

    Parameters
    ----------
    request_body:
        Validated ``RulPredictionRequest``.
    request:
        FastAPI ``Request`` used to access ``app.state``.

    Returns
    -------
    RulPredictionResponse
        Full prediction response including uncertainty bounds.
    """
    engine: PredictionEngine = request.app.state.prediction_engine

    result = await engine.predict_remaining_life(
        machine_id=request_body.machine_id,
        feature_sequence=request_body.feature_sequence,
    )

    _log.info(
        "api_rul_prediction",
        machine_id=result.machine_id,
        rul_cycles=round(result.rul_cycles, 2),
        health_label=result.health_label,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
    )

    return RulPredictionResponse(
        machine_id=result.machine_id,
        rul_cycles=result.rul_cycles,
        rul_hours=result.rul_hours,
        rul_lower_bound=result.rul_lower_bound,
        rul_upper_bound=result.rul_upper_bound,
        health_label=result.health_label,
        confidence=result.confidence,
        model_version=result.model_version,
        predicted_at=result.predicted_at,
        latency_ms=result.latency_ms,
    )


@router.post(
    "/predict/health-status",
    response_model=HealthPredictionResponse,
)
async def predict_health_status(
    request_body: HealthPredictionRequest,
    request: Request,
) -> HealthPredictionResponse:
    """Predict machine health label with class probabilities.

    Accepts a flat feature dictionary and returns a predicted health
    category (``healthy``, ``degrading``, ``critical``, or
    ``imminent_failure``) together with per-class probabilities.

    Parameters
    ----------
    request_body:
        Validated ``HealthPredictionRequest``.
    request:
        FastAPI ``Request`` used to access ``app.state``.

    Returns
    -------
    HealthPredictionResponse
        Predicted label, probabilities, and model metadata.
    """
    engine: PredictionEngine = request.app.state.prediction_engine

    result = await engine.predict_health_status(
        machine_id=request_body.machine_id,
        features=request_body.features,
    )

    _log.info(
        "api_health_prediction",
        machine_id=result.machine_id,
        health_label=result.health_label,
        model_version=result.model_version,
    )

    return HealthPredictionResponse(
        machine_id=result.machine_id,
        health_label=result.health_label,
        probabilities=result.probabilities,
        model_version=result.model_version,
        predicted_at=result.predicted_at,
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
async def predict_batch(
    request_body: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    """Process up to 100 predictions in a single API call.

    Each item in ``items`` can be either a ``"rul"`` or ``"health"``
    prediction.  Individual item failures are captured and returned
    in-line; the batch as a whole will not fail due to a single item
    error.

    Parameters
    ----------
    request_body:
        Validated ``BatchPredictionRequest`` (max 100 items).
    request:
        FastAPI ``Request`` used to access ``app.state``.

    Returns
    -------
    BatchPredictionResponse
        Results list in the same order as the input items.
    """
    engine: PredictionEngine = request.app.state.prediction_engine
    t0 = time.monotonic()

    engine_items: list[EngineBatchItem] = [
        EngineBatchItem(
            request_id=item.request_id,
            prediction_type=item.prediction_type,
            machine_id=item.machine_id,
            feature_sequence=item.feature_sequence,
            features=item.features,
        )
        for item in request_body.items
    ]

    batch_results = await engine.predict_batch(engine_items)

    total_latency_ms = int((time.monotonic() - t0) * 1000)
    n_success = sum(1 for r in batch_results if r.success)

    _log.info(
        "api_batch_prediction",
        batch_size=len(batch_results),
        n_success=n_success,
        n_failed=len(batch_results) - n_success,
        total_latency_ms=total_latency_ms,
    )

    results_list: list[dict[str, object]] = [
        {
            "request_id": r.request_id,
            "success": r.success,
            "result": r.result if r.success else None,
            "error": r.error if not r.success else None,
        }
        for r in batch_results
    ]

    return BatchPredictionResponse(
        results=results_list,
        batch_size=len(results_list),
        total_latency_ms=total_latency_ms,
    )
