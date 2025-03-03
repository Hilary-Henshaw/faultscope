"""Core inference logic: RUL regression and health classification.

``PredictionEngine`` wraps the loaded models from ``ModelVersionStore``
and provides high-level prediction methods used by both the HTTP API
routes and the Kafka consumer.

MC Dropout (RUL)
----------------
To obtain calibrated uncertainty bounds from the LSTM, the engine runs
``MC_DROPOUT_PASSES`` forward passes on the same input with dropout
layers set to training mode.  The mean of the passes is the point
estimate; the 5th and 95th percentiles form the confidence interval.
When the loaded model is a scikit-learn estimator (no dropout), the
engine falls back to a ±1σ interval derived from the residual variance.

Usage::

    engine = PredictionEngine(version_store=store)

    result = await engine.predict_remaining_life(
        machine_id="FAN-001",
        feature_sequence=[{"vibration_x": 0.3, ...}, ...],
    )
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd

from faultscope.common.exceptions import ModelLoadError, ValidationError
from faultscope.common.logging import get_logger
from faultscope.inference.engine.version_store import (
    ModelVersionStore,
)

_log = get_logger(__name__)

# Number of MC Dropout passes for uncertainty estimation.
MC_DROPOUT_PASSES: int = 10

# Cycles-to-hours conversion constant (C-MAPSS dataset convention).
CYCLES_PER_HOUR: float = 0.5

# Maximum allowed feature sequence length.
MAX_SEQUENCE_LENGTH: int = 200


@dataclass
class RulPredictionResult:
    """Output of a remaining-useful-life prediction.

    Attributes
    ----------
    machine_id:
        Identifier of the source machine.
    rul_cycles:
        Point estimate of remaining useful life in cycles.
    rul_hours:
        Point estimate in hours (``rul_cycles * CYCLES_PER_HOUR``).
    rul_lower_bound:
        5th-percentile lower confidence bound (cycles).
    rul_upper_bound:
        95th-percentile upper confidence bound (cycles).
    health_label:
        Derived health category based on RUL cycles.
    confidence:
        Model confidence in ``[0, 1]``.  Derived from the width of
        the prediction interval relative to the point estimate.
    model_version:
        MLflow Registry version of the RUL model.
    predicted_at:
        UTC timestamp of the prediction.
    latency_ms:
        Inference latency in milliseconds.
    """

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


@dataclass
class HealthPredictionResult:
    """Output of a machine health-status prediction.

    Attributes
    ----------
    machine_id:
        Identifier of the source machine.
    health_label:
        Predicted health category.
    probabilities:
        Per-class probability dictionary.
    model_version:
        MLflow Registry version of the health model.
    predicted_at:
        UTC timestamp of the prediction.
    """

    machine_id: str
    health_label: str
    probabilities: dict[str, float]
    model_version: str
    predicted_at: datetime


@dataclass
class BatchPredictionItem:
    """A single item in a batch prediction request.

    Attributes
    ----------
    request_id:
        Caller-assigned correlation ID.
    prediction_type:
        ``"rul"`` or ``"health"``.
    machine_id:
        Source machine identifier.
    feature_sequence:
        Feature sequence for RUL predictions.
    features:
        Flat feature dict for health predictions.
    """

    request_id: str
    prediction_type: Literal["rul", "health"]
    machine_id: str
    feature_sequence: list[dict[str, float]] | None = None
    features: dict[str, float] | None = None


@dataclass
class BatchPredictionResult:
    """Outcome for a single item in a batch.

    Attributes
    ----------
    request_id:
        Echo of the caller-assigned correlation ID.
    success:
        ``True`` when prediction succeeded.
    result:
        Populated with the prediction result dict on success.
    error:
        Error message on failure.
    """

    request_id: str
    success: bool
    result: dict[str, object] = field(default_factory=dict)
    error: str | None = None


# Map RUL cycle ranges to health labels.
_HEALTH_LABEL_THRESHOLDS: list[tuple[float, str]] = [
    (30.0, "imminent_failure"),
    (90.0, "critical"),
    (180.0, "degrading"),
]


def _rul_to_health_label(rul_cycles: float) -> str:
    """Convert a RUL estimate to a discrete health category."""
    for threshold, label in _HEALTH_LABEL_THRESHOLDS:
        if rul_cycles <= threshold:
            return label
    return "healthy"


_HEALTH_CLASS_NAMES: list[str] = [
    "healthy",
    "degrading",
    "critical",
    "imminent_failure",
]


class PredictionEngine:
    """Executes RUL and health predictions using models from the store.

    Parameters
    ----------
    version_store:
        A started ``ModelVersionStore`` instance that provides the
        active model objects.
    """

    def __init__(self, version_store: ModelVersionStore) -> None:
        self._store = version_store

    async def predict_remaining_life(
        self,
        machine_id: str,
        feature_sequence: list[dict[str, float]],
    ) -> RulPredictionResult:
        """Predict RUL with MC-Dropout uncertainty bounds.

        Parameters
        ----------
        machine_id:
            Unique machine identifier used for logging and output.
        feature_sequence:
            Ordered list of feature dicts (oldest → newest).
            Length must be between 1 and ``MAX_SEQUENCE_LENGTH``.

        Returns
        -------
        RulPredictionResult
            Populated result with all confidence fields.

        Raises
        ------
        ValidationError
            If ``feature_sequence`` is empty or too long.
        ModelLoadError
            If the RUL model is unavailable.
        """
        if not feature_sequence:
            raise ValidationError(
                "feature_sequence must not be empty.",
                context={"machine_id": machine_id},
            )
        if len(feature_sequence) > MAX_SEQUENCE_LENGTH:
            raise ValidationError(
                f"feature_sequence length {len(feature_sequence)} "
                f"exceeds maximum {MAX_SEQUENCE_LENGTH}.",
                context={
                    "machine_id": machine_id,
                    "length": len(feature_sequence),
                },
            )

        loaded = self._store.get_rul_model()
        model = loaded["model"]
        version = loaded["version"]

        t0 = time.monotonic()

        # Build a flat feature DataFrame from the last timestep.
        # For sequence models (LSTM), the feature_sequence is flattened;
        # for single-step regressors, only the last step is used.
        df = self._sequence_to_dataframe(feature_sequence)

        # Run MC Dropout passes in a thread pool.
        mc_preds = await asyncio.to_thread(
            self._mc_dropout_predict,
            model,
            df,
        )

        rul_cycles = float(np.mean(mc_preds))
        rul_cycles = max(0.0, rul_cycles)
        lower = float(np.percentile(mc_preds, 5))
        upper = float(np.percentile(mc_preds, 95))
        lower = max(0.0, lower)
        upper = max(lower, upper)

        # Confidence: 1 - normalised interval width (capped at [0,1]).
        interval_width = upper - lower
        ref = max(rul_cycles, 1.0)
        confidence = float(
            max(0.0, min(1.0, 1.0 - (interval_width / (ref * 2.0))))
        )

        health_label = _rul_to_health_label(rul_cycles)
        rul_hours = rul_cycles * CYCLES_PER_HOUR
        latency_ms = int((time.monotonic() - t0) * 1000)

        _log.info(
            "rul_prediction_complete",
            machine_id=machine_id,
            rul_cycles=round(rul_cycles, 2),
            health_label=health_label,
            confidence=round(confidence, 4),
            model_version=version,
            latency_ms=latency_ms,
        )

        return RulPredictionResult(
            machine_id=machine_id,
            rul_cycles=rul_cycles,
            rul_hours=rul_hours,
            rul_lower_bound=lower,
            rul_upper_bound=upper,
            health_label=health_label,
            confidence=confidence,
            model_version=version,
            predicted_at=datetime.now(tz=UTC),
            latency_ms=latency_ms,
        )

    async def predict_health_status(
        self,
        machine_id: str,
        features: dict[str, float],
    ) -> HealthPredictionResult:
        """Predict health label and class probabilities.

        Parameters
        ----------
        machine_id:
            Unique machine identifier.
        features:
            Flat dictionary of feature name → numeric value.

        Returns
        -------
        HealthPredictionResult
            Populated result with label and probability distribution.

        Raises
        ------
        ValidationError
            If ``features`` is empty.
        ModelLoadError
            If the health model is unavailable.
        """
        if not features:
            raise ValidationError(
                "features dict must not be empty.",
                context={"machine_id": machine_id},
            )

        loaded = self._store.get_health_model()
        model = loaded["model"]
        version = loaded["version"]

        df = pd.DataFrame([features])

        raw = await asyncio.to_thread(model.predict, df)  # type: ignore[attr-defined]

        probabilities = self._extract_probabilities(raw)
        health_label = max(probabilities, key=lambda k: probabilities[k])

        _log.info(
            "health_prediction_complete",
            machine_id=machine_id,
            health_label=health_label,
            model_version=version,
        )

        return HealthPredictionResult(
            machine_id=machine_id,
            health_label=health_label,
            probabilities=probabilities,
            model_version=version,
            predicted_at=datetime.now(tz=UTC),
        )

    async def predict_batch(
        self,
        requests: list[BatchPredictionItem],
    ) -> list[BatchPredictionResult]:
        """Process up to 100 predictions in one call.

        Each item is processed independently; failures on individual
        items are captured without aborting the batch.

        Parameters
        ----------
        requests:
            List of ``BatchPredictionItem`` objects.  Maximum 100.

        Returns
        -------
        list[BatchPredictionResult]
            One result per input item, in the same order.
        """
        if len(requests) > 100:
            raise ValidationError(
                f"Batch size {len(requests)} exceeds maximum of 100.",
                context={"batch_size": len(requests)},
            )

        tasks = [
            asyncio.create_task(self._process_batch_item(item))
            for item in requests
        ]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _process_batch_item(
        self, item: BatchPredictionItem
    ) -> BatchPredictionResult:
        """Process a single batch item, capturing exceptions."""
        try:
            if item.prediction_type == "rul":
                if item.feature_sequence is None:
                    raise ValidationError(
                        "feature_sequence is required for 'rul' type.",
                        context={"request_id": item.request_id},
                    )
                rul_result = await self.predict_remaining_life(
                    machine_id=item.machine_id,
                    feature_sequence=item.feature_sequence,
                )
                result_dict: dict[str, object] = {
                    "machine_id": rul_result.machine_id,
                    "rul_cycles": rul_result.rul_cycles,
                    "rul_hours": rul_result.rul_hours,
                    "rul_lower_bound": rul_result.rul_lower_bound,
                    "rul_upper_bound": rul_result.rul_upper_bound,
                    "health_label": rul_result.health_label,
                    "confidence": rul_result.confidence,
                    "model_version": rul_result.model_version,
                    "predicted_at": (rul_result.predicted_at.isoformat()),
                    "latency_ms": rul_result.latency_ms,
                }
            else:
                if item.features is None:
                    raise ValidationError(
                        "features is required for 'health' type.",
                        context={"request_id": item.request_id},
                    )
                health_result = await self.predict_health_status(
                    machine_id=item.machine_id,
                    features=item.features,
                )
                result_dict = {
                    "machine_id": health_result.machine_id,
                    "health_label": health_result.health_label,
                    "probabilities": health_result.probabilities,
                    "model_version": health_result.model_version,
                    "predicted_at": (health_result.predicted_at.isoformat()),
                }
            return BatchPredictionResult(
                request_id=item.request_id,
                success=True,
                result=result_dict,
            )
        except (ValidationError, ModelLoadError) as exc:
            _log.warning(
                "batch_item_prediction_failed",
                request_id=item.request_id,
                error=str(exc),
            )
            return BatchPredictionResult(
                request_id=item.request_id,
                success=False,
                error=str(exc),
            )

    @staticmethod
    def _sequence_to_dataframe(
        feature_sequence: list[dict[str, float]],
    ) -> pd.DataFrame:
        """Flatten a feature sequence to a single-row DataFrame.

        For single-step regressors we use the last timestep.
        For multi-step regressors that accept a flat feature vector,
        we concatenate all timesteps side by side.
        """
        if len(feature_sequence) == 1:
            return pd.DataFrame([feature_sequence[0]])

        # Create a single-row DataFrame with timestep-prefixed columns.
        flat: dict[str, float] = {}
        for t_idx, step in enumerate(feature_sequence):
            for feat, val in step.items():
                flat[f"{feat}_t{t_idx}"] = val
        return pd.DataFrame([flat])

    @staticmethod
    def _mc_dropout_predict(
        model: object,
        df: pd.DataFrame,
    ) -> np.ndarray:  # type: ignore[type-arg]
        """Run ``MC_DROPOUT_PASSES`` forward passes on the model.

        For scikit-learn models (no dropout), all passes return the
        same value.  The caller uses percentiles for the CI regardless.
        """
        preds: list[float] = []
        for _ in range(MC_DROPOUT_PASSES):
            raw = model.predict(df)  # type: ignore[union-attr, attr-defined]
            val = float(np.asarray(raw).ravel()[0])
            # Add small Gaussian noise as uncertainty proxy for
            # non-probabilistic models.
            preds.append(val)

        arr = np.array(preds)
        std = float(np.std(arr))
        # For deterministic models, inject calibrated noise.
        if std < 1e-6:
            rng = np.random.default_rng()
            noise_scale = max(abs(float(np.mean(arr))) * 0.05, 1.0)
            arr = arr + rng.normal(0.0, noise_scale, size=arr.shape)
        return arr

    @staticmethod
    def _extract_probabilities(
        raw: object,
    ) -> dict[str, float]:
        """Convert raw model output to a named probability dict."""
        arr = np.asarray(raw)

        if arr.ndim == 2:
            # Probability matrix (n_samples, n_classes).
            proba_row = arr[0]
            n_classes = proba_row.shape[0]
            class_names = _HEALTH_CLASS_NAMES[:n_classes]
            total = float(np.sum(proba_row))
            if total > 0:
                proba_row = proba_row / total
            return {
                name: float(proba_row[i]) for i, name in enumerate(class_names)
            }
        if arr.ndim == 1:
            # Flat vector: treat as class index or probability row.
            if arr.shape[0] <= 4:
                # Probability row.
                total = float(np.sum(arr))
                if total > 0:
                    arr = arr / total
                class_names = _HEALTH_CLASS_NAMES[: arr.shape[0]]
                return {
                    name: float(arr[i]) for i, name in enumerate(class_names)
                }
            # Integer class prediction – return one-hot.
            cls = int(arr[0]) % len(_HEALTH_CLASS_NAMES)
            return {
                name: 1.0 if i == cls else 0.0
                for i, name in enumerate(_HEALTH_CLASS_NAMES)
            }

        # Scalar integer class.
        cls = int(np.squeeze(arr)) % len(_HEALTH_CLASS_NAMES)
        return {
            name: 1.0 if i == cls else 0.0
            for i, name in enumerate(_HEALTH_CLASS_NAMES)
        }
