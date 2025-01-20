"""Canonical Pydantic v2 message schemas for FaultScope Kafka topics.

These models define the data contracts between every service in the
pipeline.  Any change to a schema must be backward-compatible or
accompanied by a consumer-group migration.

Serialisation notes
-------------------
- All ``datetime`` fields are serialised as ISO-8601 strings with UTC
  timezone information (``Z`` suffix) when converting to JSON.
- The ``model_config`` on each model sets ``populate_by_name=True`` so
  that field aliases and Python names are both accepted on input.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


def _serialize_dt(dt: datetime) -> str:
    """Serialise a ``datetime`` to an ISO-8601 UTC string.

    Parameters
    ----------
    dt:
        The datetime to serialise.  If it is naive it is treated as UTC.

    Returns
    -------
    str
        ISO-8601 string ending in ``Z``, e.g.
        ``"2024-03-15T12:00:00.000000Z"``.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


class SensorReading(BaseModel):
    """Raw sensor reading emitted by the ingestion service.

    One message is published per poll cycle per machine.

    Attributes
    ----------
    machine_id:
        Unique identifier of the source machine (e.g. ``"FAN-001"``).
    recorded_at:
        Wall-clock timestamp at which the reading was captured.
    cycle:
        Optional monotonically increasing operation cycle counter.
        ``None`` for equipment that does not expose cycle counts.
    readings:
        Mapping of sensor name to numeric value, e.g.
        ``{"vibration_x": 0.12, "temperature": 74.3}``.
    operational:
        Mapping of operational setting name to value, e.g.
        ``{"speed_rpm": 1500.0, "load_pct": 0.8}``.
    """

    model_config = ConfigDict(populate_by_name=True)

    machine_id: str = Field(min_length=1)
    recorded_at: datetime
    cycle: int | None = None
    readings: dict[str, float]
    operational: dict[str, float] = {}

    @field_serializer("recorded_at")
    def _ser_recorded_at(self, dt: datetime) -> str:
        """Serialise ``recorded_at`` to ISO-8601 UTC string."""
        return _serialize_dt(dt)


class ComputedFeatures(BaseModel):
    """Feature vector produced by the streaming feature-engineering service.

    Attributes
    ----------
    machine_id:
        Source machine identifier.
    computed_at:
        Timestamp when the feature computation completed.
    window_s:
        Duration in seconds of the rolling window used to compute
        aggregated features.
    temporal:
        Time-domain features (mean, std, RMS, kurtosis, …).
    spectral:
        Frequency-domain features (dominant frequency, spectral
        entropy, …).
    correlation:
        Cross-sensor correlation coefficients.
    feature_version:
        Version tag of the feature-engineering logic, enabling
        downstream models to reject incompatible feature sets.
    """

    model_config = ConfigDict(populate_by_name=True)

    machine_id: str = Field(min_length=1)
    computed_at: datetime
    window_s: int
    temporal: dict[str, float] = {}
    spectral: dict[str, float] = {}
    correlation: dict[str, float] = {}
    feature_version: str = "v1"

    @field_serializer("computed_at")
    def _ser_computed_at(self, dt: datetime) -> str:
        """Serialise ``computed_at`` to ISO-8601 UTC string."""
        return _serialize_dt(dt)


class RulPrediction(BaseModel):
    """Remaining useful life prediction emitted by the inference service.

    Attributes
    ----------
    machine_id:
        Source machine identifier.
    predicted_at:
        Timestamp when the prediction was generated.
    rul_cycles:
        Point estimate of remaining useful life in cycles.
    rul_hours:
        Point estimate of remaining useful life in hours.
    rul_lower_bound:
        Lower bound of the prediction confidence interval (cycles).
    rul_upper_bound:
        Upper bound of the prediction confidence interval (cycles).
    health_label:
        Discrete health classification.
    health_probabilities:
        Softmax probability per health class, e.g.
        ``{"healthy": 0.02, "critical": 0.97, ...}``.
    anomaly_score:
        Continuous anomaly score in ``[0, 1]``.  Higher values
        indicate greater deviation from nominal behaviour.
    confidence:
        Overall model confidence in ``[0, 1]``.
    rul_model_version:
        Semantic version of the RUL regression model.
    health_model_version:
        Semantic version of the health-classification model.
    latency_ms:
        End-to-end inference latency in milliseconds, for SLA
        monitoring.
    """

    model_config = ConfigDict(populate_by_name=True)

    machine_id: str = Field(min_length=1)
    predicted_at: datetime
    rul_cycles: float
    rul_hours: float
    rul_lower_bound: float
    rul_upper_bound: float
    health_label: Literal[
        "healthy",
        "degrading",
        "critical",
        "imminent_failure",
    ]
    health_probabilities: dict[str, float] = {}
    anomaly_score: float
    confidence: float
    rul_model_version: str
    health_model_version: str
    latency_ms: int = 0

    @field_serializer("predicted_at")
    def _ser_predicted_at(self, dt: datetime) -> str:
        """Serialise ``predicted_at`` to ISO-8601 UTC string."""
        return _serialize_dt(dt)
