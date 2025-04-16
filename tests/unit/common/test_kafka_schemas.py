"""Unit tests for SensorReading and RulPrediction Kafka schemas.

Tests cover serialization round-trips, field validation, and
Pydantic v2 constraint enforcement.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from faultscope.common.kafka.schemas import RulPrediction, SensorReading


@pytest.mark.unit
class TestSensorReadingSchema:
    """Tests for SensorReading Pydantic schema."""

    def test_valid_reading_serializes_to_json(
        self,
        sample_sensor_reading: SensorReading,
    ) -> None:
        """model_dump_json must produce valid JSON without raising."""
        json_str = sample_sensor_reading.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["machine_id"] == "ENG_001"

    def test_round_trip_json_preserves_all_fields(
        self,
        sample_sensor_reading: SensorReading,
    ) -> None:
        """Deserializing from model_dump_json must reproduce the original."""
        json_str = sample_sensor_reading.model_dump_json()
        reconstructed = SensorReading.model_validate_json(json_str)
        assert reconstructed.machine_id == sample_sensor_reading.machine_id
        assert reconstructed.cycle == sample_sensor_reading.cycle
        assert set(reconstructed.readings.keys()) == set(
            sample_sensor_reading.readings.keys()
        )
        for key in sample_sensor_reading.readings:
            assert reconstructed.readings[key] == pytest.approx(
                sample_sensor_reading.readings[key], rel=1e-9
            )

    def test_rejects_empty_machine_id(self) -> None:
        """Empty string machine_id must fail Pydantic validation."""
        with pytest.raises(ValidationError):
            SensorReading(
                machine_id="",
                recorded_at=datetime.now(tz=UTC),
                readings={"s1": 1.0},
            )

    def test_accepts_none_for_optional_cycle(self) -> None:
        """cycle field is optional and None must be accepted."""
        reading = SensorReading(
            machine_id="ENG_001",
            recorded_at=datetime.now(tz=UTC),
            cycle=None,
            readings={"s1": 1.0},
        )
        assert reading.cycle is None

    def test_recorded_at_serialized_as_utc_iso_string(
        self,
        sample_sensor_reading: SensorReading,
    ) -> None:
        """recorded_at must serialize to a UTC ISO-8601 string ending in Z."""
        data = json.loads(sample_sensor_reading.model_dump_json())
        recorded_at_str = data["recorded_at"]
        assert isinstance(recorded_at_str, str)
        # Serializer replaces +00:00 with Z suffix.
        assert recorded_at_str.endswith("Z")

    def test_readings_dict_preserved_in_round_trip(self) -> None:
        """All sensor key/value pairs survive serialization round-trip."""
        readings = {"temp": 100.5, "pressure": 14.7, "rpm": 3000.0}
        r = SensorReading(
            machine_id="ENG_001",
            recorded_at=datetime.now(tz=UTC),
            readings=readings,
        )
        reconstructed = SensorReading.model_validate_json(r.model_dump_json())
        assert reconstructed.readings == pytest.approx(readings, rel=1e-9)

    def test_operational_defaults_to_empty_dict(self) -> None:
        """omitting operational must default to an empty dict."""
        r = SensorReading(
            machine_id="ENG_001",
            recorded_at=datetime.now(tz=UTC),
            readings={"s1": 1.0},
        )
        assert r.operational == {}

    def test_populate_by_name_allows_field_name_on_input(self) -> None:
        """populate_by_name=True means python field names are accepted."""
        r = SensorReading.model_validate(
            {
                "machine_id": "ENG_001",
                "recorded_at": datetime.now(tz=UTC),
                "readings": {"s1": 1.0},
                "operational": {"speed": 1500.0},
            }
        )
        assert r.operational["speed"] == 1500.0


@pytest.mark.unit
class TestRulPredictionSchema:
    """Tests for RulPrediction Pydantic schema."""

    def test_valid_prediction_serializes_to_json(
        self,
        sample_rul_prediction: RulPrediction,
    ) -> None:
        """model_dump_json must produce valid JSON without raising."""
        json_str = sample_rul_prediction.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["machine_id"] == "ENG_001"
        assert parsed["rul_cycles"] == pytest.approx(25.0)

    def test_round_trip_json_preserves_all_fields(
        self,
        sample_rul_prediction: RulPrediction,
    ) -> None:
        """Deserializing from model_dump_json must reproduce the original."""
        json_str = sample_rul_prediction.model_dump_json()
        reconstructed = RulPrediction.model_validate_json(json_str)
        assert reconstructed.machine_id == sample_rul_prediction.machine_id
        assert reconstructed.rul_cycles == pytest.approx(
            sample_rul_prediction.rul_cycles
        )
        assert reconstructed.health_label == sample_rul_prediction.health_label
        assert reconstructed.confidence == pytest.approx(
            sample_rul_prediction.confidence
        )

    def test_rejects_invalid_health_label(self) -> None:
        """health_label must be one of the four allowed literals."""
        with pytest.raises(ValidationError):
            RulPrediction(
                machine_id="ENG_001",
                predicted_at=datetime.now(tz=UTC),
                rul_cycles=25.0,
                rul_hours=50.0,
                rul_lower_bound=15.0,
                rul_upper_bound=35.0,
                health_label="unknown_label",  # type: ignore[arg-type]
                anomaly_score=0.45,
                confidence=0.87,
                rul_model_version="v1.0.0",
                health_model_version="v1.0.0",
            )

    def test_confidence_accepted_in_0_1_range(self) -> None:
        """confidence in [0, 1] must be accepted without raising."""
        for conf in [0.0, 0.5, 1.0]:
            pred = RulPrediction(
                machine_id="ENG_001",
                predicted_at=datetime.now(tz=UTC),
                rul_cycles=25.0,
                rul_hours=50.0,
                rul_lower_bound=15.0,
                rul_upper_bound=35.0,
                health_label="healthy",
                anomaly_score=0.1,
                confidence=conf,
                rul_model_version="v1.0.0",
                health_model_version="v1.0.0",
            )
            assert pred.confidence == conf

    def test_all_four_health_labels_are_accepted(self) -> None:
        """Each of the four health labels must be a valid literal."""
        valid_labels = [
            "healthy",
            "degrading",
            "critical",
            "imminent_failure",
        ]
        for label in valid_labels:
            pred = RulPrediction(
                machine_id="ENG_001",
                predicted_at=datetime.now(tz=UTC),
                rul_cycles=25.0,
                rul_hours=50.0,
                rul_lower_bound=15.0,
                rul_upper_bound=35.0,
                health_label=label,  # type: ignore[arg-type]
                anomaly_score=0.1,
                confidence=0.9,
                rul_model_version="v1.0.0",
                health_model_version="v1.0.0",
            )
            assert pred.health_label == label

    def test_predicted_at_serialized_as_utc_iso_string(
        self,
        sample_rul_prediction: RulPrediction,
    ) -> None:
        """predicted_at must serialize to a UTC ISO string ending in Z."""
        data = json.loads(sample_rul_prediction.model_dump_json())
        assert data["predicted_at"].endswith("Z")

    def test_latency_ms_defaults_to_zero(self) -> None:
        """latency_ms has a default of 0 when not supplied."""
        pred = RulPrediction(
            machine_id="ENG_001",
            predicted_at=datetime.now(tz=UTC),
            rul_cycles=25.0,
            rul_hours=50.0,
            rul_lower_bound=15.0,
            rul_upper_bound=35.0,
            health_label="healthy",
            anomaly_score=0.1,
            confidence=0.9,
            rul_model_version="v1.0.0",
            health_model_version="v1.0.0",
        )
        assert pred.latency_ms == 0

    def test_rul_confidence_bounds_preserved(
        self,
        sample_rul_prediction: RulPrediction,
    ) -> None:
        """rul_lower_bound and rul_upper_bound survive round-trip intact."""
        json_str = sample_rul_prediction.model_dump_json()
        reconstructed = RulPrediction.model_validate_json(json_str)
        assert reconstructed.rul_lower_bound == pytest.approx(15.0)
        assert reconstructed.rul_upper_bound == pytest.approx(35.0)

    def test_health_probabilities_dict_preserved(
        self,
        sample_rul_prediction: RulPrediction,
    ) -> None:
        """health_probabilities dict survives JSON round-trip."""
        json_str = sample_rul_prediction.model_dump_json()
        r = RulPrediction.model_validate_json(json_str)
        assert r.health_probabilities["degrading"] == pytest.approx(
            0.60, rel=1e-9
        )

    def test_model_copy_update_changes_target_field_only(
        self,
        sample_rul_prediction: RulPrediction,
    ) -> None:
        """model_copy with update must change only the specified field."""
        updated = sample_rul_prediction.model_copy(update={"rul_cycles": 5.0})
        assert updated.rul_cycles == 5.0
        assert updated.machine_id == sample_rul_prediction.machine_id
        assert updated.confidence == sample_rul_prediction.confidence
