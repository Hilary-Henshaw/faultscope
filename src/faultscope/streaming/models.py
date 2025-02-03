"""Pydantic data models for the streaming service.

``SensorReading`` is the deserialized form of messages arriving on
``faultscope.sensors.readings``.  ``ComputedFeatures`` is published
to ``faultscope.features.computed`` and written to TimescaleDB.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    """A single batch of sensor measurements from one machine.

    Attributes
    ----------
    machine_id:
        Unique identifier of the source machine.
    recorded_at:
        UTC timestamp when the readings were captured at the machine.
    cycle:
        Optional operational cycle counter (used by CMAPSS-style
        datasets).
    readings:
        Mapping of sensor name to measured float value.
    operational:
        Supplementary operational settings (e.g. speed set-point),
        stored alongside the readings.
    """

    machine_id: str
    recorded_at: datetime
    cycle: int | None = None
    readings: dict[str, float]
    operational: dict[str, float] = Field(default_factory=dict)


class ComputedFeatures(BaseModel):
    """Feature vector computed by the streaming pipeline.

    Published to Kafka and persisted in the ``computed_features``
    TimescaleDB hypertable.

    Attributes
    ----------
    machine_id:
        Identifier of the machine the features describe.
    computed_at:
        UTC timestamp when features were extracted.
    window_s:
        Rolling window duration (seconds) used for temporal features.
    temporal:
        Time-domain statistical features (mean, std, rms, etc.).
    spectral:
        Frequency-domain features (dominant freq, entropy, etc.).
    correlation:
        Cross-sensor Pearson correlation coefficients.
    feature_version:
        Schema version tag for downstream compatibility checks.
    """

    machine_id: str
    computed_at: datetime
    window_s: int
    temporal: dict[str, float] = Field(default_factory=dict)
    spectral: dict[str, float] = Field(default_factory=dict)
    correlation: dict[str, float] = Field(default_factory=dict)
    feature_version: str = "v1"
