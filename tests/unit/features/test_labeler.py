"""Unit tests for RulLabeler and HealthLabeler.

Verifies RUL computation at boundary cycles, capping behaviour,
per-machine independence, and health threshold classification.
"""

from __future__ import annotations

import pandas as pd
import pytest

from faultscope.common.exceptions import ValidationError
from faultscope.features.labeler import HealthLabeler, RulLabeler

# Default health thresholds matching production config.
_DEFAULT_THRESHOLDS: dict[str, int] = {
    "healthy": 100,
    "degrading": 50,
    "critical": 20,
    "imminent_failure": 0,
}


def _make_df(
    machine_id: str,
    cycles: list[int],
) -> pd.DataFrame:
    """Create a minimal DataFrame with machine_id and cycle columns."""
    return pd.DataFrame(
        {
            "machine_id": machine_id,
            "cycle": cycles,
        }
    )


@pytest.mark.unit
class TestRulLabeler:
    """Unit tests for RulLabeler."""

    def test_rul_is_zero_at_last_cycle(self) -> None:
        """RUL for the final cycle in a machine's life must be 0."""
        labeler = RulLabeler(max_rul_cycles=125)
        df = _make_df("M001", [1, 2, 3, 4, 5])
        result = labeler.assign_rul(df)
        last_cycle_row = result[result["cycle"] == 5]
        assert int(last_cycle_row["rul_cycles"].iloc[0]) == 0

    def test_rul_equals_total_cycles_minus_one_at_first_cycle_uncapped(
        self,
    ) -> None:
        """At cycle 1 of a 5-cycle machine RUL = 4 (= 5 - 1), uncapped."""
        labeler = RulLabeler(max_rul_cycles=125)
        df = _make_df("M001", [1, 2, 3, 4, 5])
        result = labeler.assign_rul(df)
        first_row = result[result["cycle"] == 1]
        assert int(first_row["rul_cycles"].iloc[0]) == 4

    def test_rul_equals_total_cycles_at_first_cycle(self) -> None:
        """With large max_rul, first cycle equals max_cycle − cycle."""
        labeler = RulLabeler(max_rul_cycles=500)
        df = _make_df("M001", list(range(1, 101)))  # cycles 1–100
        result = labeler.assign_rul(df)
        first_row = result[result["cycle"] == 1]
        assert int(first_row["rul_cycles"].iloc[0]) == 99

    def test_rul_is_capped_at_max_value(self) -> None:
        """RUL values must not exceed max_rul_cycles."""
        labeler = RulLabeler(max_rul_cycles=10)
        # 50 cycles → raw RUL at cycle 1 is 49, but cap is 10.
        df = _make_df("M001", list(range(1, 51)))
        result = labeler.assign_rul(df)
        assert int(result["rul_cycles"].max()) == 10

    def test_rul_column_added_to_dataframe(self) -> None:
        """assign_rul must add the rul_cycles column to the DataFrame."""
        labeler = RulLabeler()
        df = _make_df("M001", [1, 2, 3])
        result = labeler.assign_rul(df)
        assert "rul_cycles" in result.columns

    def test_multiple_machines_labeled_independently(self) -> None:
        """RUL labels for each machine must be computed from its own max."""
        labeler = RulLabeler(max_rul_cycles=500)
        df = pd.concat(
            [
                _make_df("M001", [1, 2, 3]),  # 3 cycles → last RUL=0
                _make_df("M002", [1, 2, 3, 4, 5]),  # 5 cycles → last RUL=0
            ]
        )
        result = labeler.assign_rul(df)

        m1_last = result[
            (result["machine_id"] == "M001") & (result["cycle"] == 3)
        ]
        m2_last = result[
            (result["machine_id"] == "M002") & (result["cycle"] == 5)
        ]

        assert int(m1_last["rul_cycles"].iloc[0]) == 0
        assert int(m2_last["rul_cycles"].iloc[0]) == 0

        # First cycle RUL differs because each machine has its own lifecycle.
        m1_first = result[
            (result["machine_id"] == "M001") & (result["cycle"] == 1)
        ]
        m2_first = result[
            (result["machine_id"] == "M002") & (result["cycle"] == 1)
        ]
        assert int(m1_first["rul_cycles"].iloc[0]) == 2  # 3 - 1
        assert int(m2_first["rul_cycles"].iloc[0]) == 4  # 5 - 1

    def test_raises_validation_error_for_missing_columns(self) -> None:
        """assign_rul must raise ValidationError when columns are absent."""
        labeler = RulLabeler()
        bad_df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValidationError):
            labeler.assign_rul(bad_df)

    def test_raises_validation_error_for_non_positive_max_rul(self) -> None:
        """Constructor raises ValidationError for max_rul_cycles <= 0."""
        with pytest.raises(ValidationError):
            RulLabeler(max_rul_cycles=0)

    def test_rul_all_non_negative(self) -> None:
        """No RUL value must be negative."""
        labeler = RulLabeler(max_rul_cycles=125)
        df = _make_df("M001", list(range(1, 201)))
        result = labeler.assign_rul(df)
        assert (result["rul_cycles"] >= 0).all()

    def test_rul_decreases_monotonically_for_single_machine(self) -> None:
        """RUL values must be non-increasing as cycle increases."""
        labeler = RulLabeler(max_rul_cycles=125)
        df = _make_df("M001", list(range(1, 51)))
        result = labeler.assign_rul(df).sort_values("cycle")
        rul = result["rul_cycles"].to_numpy()
        # Each subsequent value must be <= the previous.
        assert all(rul[i] >= rul[i + 1] for i in range(len(rul) - 1))


@pytest.mark.unit
class TestHealthLabeler:
    """Unit tests for HealthLabeler."""

    @pytest.fixture
    def labeler(self) -> HealthLabeler:
        return HealthLabeler(thresholds=_DEFAULT_THRESHOLDS)

    def test_rul_above_100_is_healthy(self, labeler: HealthLabeler) -> None:
        """RUL >= 100 must receive the 'healthy' label."""
        df = pd.DataFrame({"rul_cycles": [100, 150, 200]})
        result = labeler.assign_health(df)
        assert (result["health_label"] == "healthy").all()

    def test_rul_between_50_and_99_is_degrading(
        self, labeler: HealthLabeler
    ) -> None:
        """RUL in [50, 99] must receive the 'degrading' label."""
        df = pd.DataFrame({"rul_cycles": [50, 75, 99]})
        result = labeler.assign_health(df)
        assert (result["health_label"] == "degrading").all()

    def test_rul_between_20_and_49_is_critical(
        self, labeler: HealthLabeler
    ) -> None:
        """RUL in [20, 49] must receive the 'critical' label."""
        df = pd.DataFrame({"rul_cycles": [20, 30, 49]})
        result = labeler.assign_health(df)
        assert (result["health_label"] == "critical").all()

    def test_rul_below_20_is_critical(self, labeler: HealthLabeler) -> None:
        """RUL in [0, 19] must receive 'imminent_failure' label."""
        df = pd.DataFrame({"rul_cycles": [0, 10, 19]})
        result = labeler.assign_health(df)
        assert (result["health_label"] == "imminent_failure").all()

    def test_rul_below_10_is_imminent_failure(
        self, labeler: HealthLabeler
    ) -> None:
        """RUL=5 must map to 'imminent_failure'."""
        df = pd.DataFrame({"rul_cycles": [5]})
        result = labeler.assign_health(df)
        assert result["health_label"].iloc[0] == "imminent_failure"

    def test_all_rows_get_a_health_label(self, labeler: HealthLabeler) -> None:
        """Every row must have a non-null health_label after assignment."""
        df = pd.DataFrame({"rul_cycles": list(range(0, 201, 10))})
        result = labeler.assign_health(df)
        assert result["health_label"].notna().all()
        valid = {"healthy", "degrading", "critical", "imminent_failure"}
        assert set(result["health_label"].unique()).issubset(valid)

    def test_raises_validation_error_for_missing_rul_column(
        self, labeler: HealthLabeler
    ) -> None:
        """assign_health raises ValidationError when rul_cycles is absent."""
        bad_df = pd.DataFrame({"foo": [1, 2, 3]})
        with pytest.raises(ValidationError):
            labeler.assign_health(bad_df)

    def test_raises_validation_error_for_missing_threshold_labels(
        self,
    ) -> None:
        """Constructor raises ValidationError for missing threshold labels."""
        incomplete = {
            "healthy": 100,
            "degrading": 50,
            # missing 'critical' and 'imminent_failure'
        }
        with pytest.raises(ValidationError):
            HealthLabeler(thresholds=incomplete)

    def test_boundary_value_at_healthy_threshold(
        self, labeler: HealthLabeler
    ) -> None:
        """RUL exactly at the healthy threshold (100) must be 'healthy'."""
        df = pd.DataFrame({"rul_cycles": [100]})
        result = labeler.assign_health(df)
        assert result["health_label"].iloc[0] == "healthy"

    def test_boundary_value_just_below_healthy_threshold(
        self, labeler: HealthLabeler
    ) -> None:
        """RUL=99 must NOT be 'healthy' (below 100 threshold)."""
        df = pd.DataFrame({"rul_cycles": [99]})
        result = labeler.assign_health(df)
        assert result["health_label"].iloc[0] != "healthy"

    def test_health_column_added_to_dataframe(
        self, labeler: HealthLabeler
    ) -> None:
        """assign_health must add the health_label column."""
        df = pd.DataFrame({"rul_cycles": [50, 100, 5]})
        result = labeler.assign_health(df)
        assert "health_label" in result.columns

    def test_assign_health_does_not_modify_original_df(
        self, labeler: HealthLabeler
    ) -> None:
        """assign_health must return a copy and not mutate the input."""
        original = pd.DataFrame({"rul_cycles": [50, 100, 5]})
        _ = labeler.assign_health(original)
        assert "health_label" not in original.columns
