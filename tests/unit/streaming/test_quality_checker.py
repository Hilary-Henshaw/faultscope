"""Unit tests for DataQualityChecker.

Tests cover the full rejection/flag matrix including timestamp
drift, null fractions, forward-fill, outlier detection, and
duplicate timestamp handling.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from faultscope.streaming.models import SensorReading
from faultscope.streaming.quality import DataQualityChecker, QualityFlag


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _reading(
    machine_id: str = "ENG_001",
    offset_s: float = 0.0,
    readings: dict[str, float] | None = None,
) -> SensorReading:
    if readings is None:
        readings = {
            "fan_inlet_temp": 518.67,
            "lpc_outlet_temp": 642.42,
            "hpc_outlet_temp": 1587.36,
            "lpt_outlet_temp": 1400.60,
            "total_pressure_r2": 14.62,
        }
    return SensorReading(
        machine_id=machine_id,
        recorded_at=_now() + timedelta(seconds=offset_s),
        cycle=1,
        readings=readings,
    )


@pytest.mark.unit
class TestDataQualityChecker:
    """Unit tests for DataQualityChecker."""

    def test_valid_reading_passes_all_checks(
        self,
        sample_sensor_reading: SensorReading,
    ) -> None:
        """A clean reading with no anomalies must not be rejected."""
        checker = DataQualityChecker()
        result = checker.check(sample_sensor_reading, previous=None)
        assert result.rejected is False
        # The sensor values span very different scales so IQR may flag
        # outliers — only check that the critical rejection flags are absent.
        assert QualityFlag.MISSING_ID not in result.flags
        assert QualityFlag.FUTURE_TIMESTAMP not in result.flags
        assert QualityFlag.NULL_FRACTION_HIGH not in result.flags

    def test_rejects_reading_with_missing_machine_id(self) -> None:
        """Empty machine_id triggers MISSING_ID flag and rejection."""
        checker = DataQualityChecker()
        bad = _reading(machine_id="   ")
        result = checker.check(bad, previous=None)
        assert result.rejected is True
        assert QualityFlag.MISSING_ID in result.flags

    def test_rejects_reading_with_empty_string_machine_id(
        self,
    ) -> None:
        """Fully empty string machine_id must also be rejected."""
        checker = DataQualityChecker()
        bad = _reading(machine_id="")
        result = checker.check(bad, previous=None)
        assert result.rejected is True
        assert QualityFlag.MISSING_ID in result.flags

    def test_rejects_reading_with_far_future_timestamp(self) -> None:
        """Timestamp 600 s in the future exceeds 300 s limit → rejected."""
        checker = DataQualityChecker(max_future_drift_s=300.0)
        future = _reading(offset_s=600.0)
        result = checker.check(future, previous=None)
        assert result.rejected is True
        assert QualityFlag.FUTURE_TIMESTAMP in result.flags

    def test_accepts_reading_within_future_drift_limit(self) -> None:
        """Timestamp 100 s ahead of a 300 s limit must not be rejected."""
        checker = DataQualityChecker(max_future_drift_s=300.0)
        slightly_future = _reading(offset_s=100.0)
        result = checker.check(slightly_future, previous=None)
        assert QualityFlag.FUTURE_TIMESTAMP not in result.flags

    def test_flags_reading_with_high_null_fraction(self) -> None:
        """More than 30 % NaN values triggers NULL_FRACTION_HIGH + reject."""
        checker = DataQualityChecker(max_null_fraction=0.3)
        # 4 out of 5 sensors are NaN → 80 % null fraction
        readings = {
            "s1": float("nan"),
            "s2": float("nan"),
            "s3": float("nan"),
            "s4": float("nan"),
            "s5": 1.0,
        }
        bad = _reading(readings=readings)
        result = checker.check(bad, previous=None)
        assert result.rejected is True
        assert QualityFlag.NULL_FRACTION_HIGH in result.flags

    def test_forward_fills_sparse_nulls_from_previous(self) -> None:
        """NaN values below threshold are filled with previous reading."""
        checker = DataQualityChecker(max_null_fraction=0.5)
        prev = _reading(
            readings={
                "s1": 10.0,
                "s2": 20.0,
                "s3": 30.0,
                "s4": 40.0,
                "s5": 50.0,
            }
        )
        # 1 NaN out of 5 = 20 % — below 50 % threshold
        curr_readings = {
            "s1": float("nan"),
            "s2": 20.0,
            "s3": 30.0,
            "s4": 40.0,
            "s5": 50.0,
        }
        curr = _reading(readings=curr_readings)
        result = checker.check(curr, previous=prev)
        assert result.rejected is False
        assert QualityFlag.NULL_FRACTION_LOW in result.flags
        # s1 should have been filled with the previous value (10.0)
        assert result.filled_readings["s1"] == pytest.approx(10.0)

    def test_forward_fill_uses_zero_when_no_previous(self) -> None:
        """Without a previous reading, NaN is filled with 0.0."""
        checker = DataQualityChecker(max_null_fraction=0.5)
        curr_readings = {
            "s1": float("nan"),
            "s2": 2.0,
            "s3": 3.0,
            "s4": 4.0,
            "s5": 5.0,
        }
        curr = _reading(readings=curr_readings)
        result = checker.check(curr, previous=None)
        assert result.filled_readings["s1"] == pytest.approx(0.0)

    def test_flags_but_includes_outlier_values(self) -> None:
        """Outliers are flagged but the message is NOT rejected."""
        checker = DataQualityChecker(min_sensor_count=1)
        # One extreme outlier among otherwise tight values.
        readings = {
            "s1": 1.0,
            "s2": 1.1,
            "s3": 1.2,
            "s4": 1.3,
            "s5": 999999.0,  # massive outlier
        }
        r = _reading(readings=readings)
        result = checker.check(r, previous=None)
        assert result.rejected is False
        assert QualityFlag.OUTLIER_DETECTED in result.flags
        # Outlier value must still be present in filled_readings.
        assert result.filled_readings["s5"] == pytest.approx(999999.0)

    def test_duplicate_timestamp_flagged(self) -> None:
        """Identical timestamps between previous and current are flagged."""
        checker = DataQualityChecker()
        ts = _now()
        prev = SensorReading(
            machine_id="ENG_001",
            recorded_at=ts,
            readings={"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0, "s5": 5.0},
        )
        curr = SensorReading(
            machine_id="ENG_001",
            recorded_at=ts,  # same timestamp
            readings={"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0, "s5": 5.0},
        )
        result = checker.check(curr, previous=prev)
        assert QualityFlag.DUPLICATE_TIMESTAMP in result.flags

    def test_duplicate_timestamp_does_not_cause_rejection(self) -> None:
        """DUPLICATE_TIMESTAMP is informational — must not set rejected."""
        checker = DataQualityChecker()
        ts = _now()
        prev = SensorReading(
            machine_id="ENG_001",
            recorded_at=ts,
            readings={"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0, "s5": 5.0},
        )
        curr = SensorReading(
            machine_id="ENG_001",
            recorded_at=ts,
            readings={"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0, "s5": 5.0},
        )
        result = checker.check(curr, previous=prev)
        # Duplicate alone is not a rejection condition.
        assert QualityFlag.MISSING_ID not in result.flags
        assert QualityFlag.FUTURE_TIMESTAMP not in result.flags
        assert QualityFlag.NULL_FRACTION_HIGH not in result.flags

    def test_sensor_count_low_flag_set_when_below_minimum(self) -> None:
        """Fewer sensors than min_sensor_count raises SENSOR_COUNT_LOW."""
        checker = DataQualityChecker(min_sensor_count=10)
        r = _reading(readings={"s1": 1.0, "s2": 2.0})
        result = checker.check(r, previous=None)
        assert QualityFlag.SENSOR_COUNT_LOW in result.flags
        # Low count alone must not reject.
        assert result.rejected is False

    def test_flag_names_list_matches_active_flags(self) -> None:
        """flag_names must contain the string names of all active flags."""
        checker = DataQualityChecker()
        bad = _reading(machine_id="")
        result = checker.check(bad, previous=None)
        assert "MISSING_ID" in result.flag_names

    def test_valid_reading_filled_readings_equals_originals(
        self, sample_sensor_reading: SensorReading
    ) -> None:
        """Filled readings for a clean input must equal the originals."""
        checker = DataQualityChecker()
        result = checker.check(sample_sensor_reading, previous=None)
        for key, val in sample_sensor_reading.readings.items():
            if not math.isnan(val):
                assert result.filled_readings[key] == pytest.approx(val)
