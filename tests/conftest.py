"""Root-level pytest fixtures shared across the entire test suite.

Provides fully populated ``SensorReading`` and ``RulPrediction``
instances that represent typical turbofan engine data drawn from the
CMAPSS dataset conventions used throughout FaultScope.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from faultscope.common.kafka.schemas import RulPrediction, SensorReading


@pytest.fixture
def sample_sensor_reading() -> SensorReading:
    """Valid turbofan sensor reading for testing."""
    return SensorReading(
        machine_id="ENG_001",
        recorded_at=datetime.now(tz=UTC),
        cycle=100,
        readings={
            "fan_inlet_temp": 518.67,
            "lpc_outlet_temp": 642.42,
            "hpc_outlet_temp": 1587.36,
            "lpt_outlet_temp": 1400.60,
            "total_pressure_r2": 14.62,
            "vibration_rms": 0.23,
            "bearing_temp": 65.2,
        },
        operational={
            "op_setting_1": -0.0007,
            "op_setting_2": -0.0004,
        },
    )


@pytest.fixture
def sample_rul_prediction() -> RulPrediction:
    """Valid RUL prediction for testing alerting rules."""
    return RulPrediction(
        machine_id="ENG_001",
        predicted_at=datetime.now(tz=UTC),
        rul_cycles=25.0,
        rul_hours=50.0,
        rul_lower_bound=15.0,
        rul_upper_bound=35.0,
        health_label="degrading",
        health_probabilities={
            "healthy": 0.05,
            "degrading": 0.60,
            "critical": 0.30,
            "imminent_failure": 0.05,
        },
        anomaly_score=0.45,
        confidence=0.87,
        rul_model_version="v1.0.0",
        health_model_version="v1.0.0",
    )
