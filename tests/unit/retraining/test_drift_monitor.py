"""Unit tests for DriftMonitor.

Tests cover KS-test data drift detection, concept drift via t-test,
edge cases (identical distributions, empty inputs), and report fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from faultscope.retraining.drift import DriftMonitor, DriftReport


def _gauss(
    mean: float = 0.0,
    std: float = 1.0,
    n: int = 500,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=n)


def _df(
    cols: dict[str, np.ndarray],
) -> pd.DataFrame:
    return pd.DataFrame(cols)


@pytest.mark.unit
class TestDriftMonitor:
    """Unit tests for DriftMonitor."""

    # ── Data drift ────────────────────────────────────────────────────

    def test_no_drift_detected_for_identical_distributions(self) -> None:
        """KS test on two samples from the exact same distribution
        must not flag drift (p-value should be high)."""
        monitor = DriftMonitor(ks_p_threshold=0.05)
        data = _gauss(mean=0.0, std=1.0, n=500, seed=0)
        ref = _df({"feature_a": data})
        # Use a different random seed but same distribution.
        cur_data = _gauss(mean=0.0, std=1.0, n=500, seed=1)
        cur = _df({"feature_a": cur_data})

        report = monitor.detect_data_drift(ref, cur, ["feature_a"])

        assert isinstance(report, DriftReport)
        assert report.detected is False
        assert report.recommendation in ("ok", "monitor")
        assert "feature_a" not in report.affected_features

    def test_drift_detected_for_shifted_distribution(self) -> None:
        """Mean shift of 3 standard deviations should trigger drift."""
        monitor = DriftMonitor(ks_p_threshold=0.05)
        ref = _df({"feature_a": _gauss(mean=0.0, std=1.0, n=500, seed=10)})
        cur = _df({"feature_a": _gauss(mean=3.0, std=1.0, n=500, seed=11)})

        report = monitor.detect_data_drift(ref, cur, ["feature_a"])

        assert report.detected is True
        assert "feature_a" in report.affected_features
        assert report.recommendation == "retrain"

    def test_ks_statistics_present_for_all_tested_features(self) -> None:
        """DriftReport must contain KS statistic for every tested feature."""
        monitor = DriftMonitor(ks_p_threshold=0.05)
        cols = ["feat_x", "feat_y", "feat_z"]
        ref_data = {c: _gauss(seed=i) for i, c in enumerate(cols)}
        cur_data = {c: _gauss(seed=i + 100) for i, c in enumerate(cols)}
        ref = _df(ref_data)
        cur = _df(cur_data)

        report = monitor.detect_data_drift(ref, cur, cols)

        for col in cols:
            assert col in report.ks_statistics
            assert col in report.p_values

    def test_most_drifted_features_listed_in_report(self) -> None:
        """Drifted features must appear in affected_features."""
        monitor = DriftMonitor(ks_p_threshold=0.05)
        # feature_b has large shift; feature_a has no shift.
        ref = _df(
            {
                "feature_a": _gauss(0.0, 1.0, seed=0),
                "feature_b": _gauss(0.0, 1.0, seed=1),
            }
        )
        cur = _df(
            {
                "feature_a": _gauss(0.0, 1.0, seed=2),
                "feature_b": _gauss(5.0, 1.0, seed=3),  # big shift
            }
        )
        report = monitor.detect_data_drift(
            ref, cur, ["feature_a", "feature_b"]
        )
        assert "feature_b" in report.affected_features

    def test_drift_type_is_covariate_for_data_drift(self) -> None:
        """detect_data_drift must set drift_type='covariate'."""
        monitor = DriftMonitor()
        ref = _df({"f": _gauss(seed=0)})
        cur = _df({"f": _gauss(mean=4.0, seed=1)})
        report = monitor.detect_data_drift(ref, cur, ["f"])
        assert report.drift_type == "covariate"

    def test_raises_when_column_missing_from_reference(self) -> None:
        """detect_data_drift raises ValueError for missing ref column."""
        monitor = DriftMonitor()
        ref = _df({"a": _gauss(seed=0)})
        cur = _df({"a": _gauss(seed=1), "b": _gauss(seed=2)})
        with pytest.raises(ValueError, match="Reference"):
            monitor.detect_data_drift(ref, cur, ["a", "b"])

    def test_raises_when_column_missing_from_current(self) -> None:
        """detect_data_drift raises ValueError for missing current column."""
        monitor = DriftMonitor()
        ref = _df({"a": _gauss(seed=0), "b": _gauss(seed=1)})
        cur = _df({"a": _gauss(seed=2)})
        with pytest.raises(ValueError, match="Current"):
            monitor.detect_data_drift(ref, cur, ["a", "b"])

    def test_raises_when_reference_dataframe_is_empty(self) -> None:
        """Empty reference DataFrame must raise ValueError."""
        monitor = DriftMonitor()
        ref = pd.DataFrame({"a": []})
        cur = _df({"a": _gauss(seed=0)})
        with pytest.raises(ValueError, match="empty"):
            monitor.detect_data_drift(ref, cur, ["a"])

    def test_raises_when_current_dataframe_is_empty(self) -> None:
        """Empty current DataFrame must raise ValueError."""
        monitor = DriftMonitor()
        ref = _df({"a": _gauss(seed=0)})
        cur = pd.DataFrame({"a": []})
        with pytest.raises(ValueError, match="empty"):
            monitor.detect_data_drift(ref, cur, ["a"])

    def test_error_increase_is_none_for_data_drift_report(self) -> None:
        """Data drift reports must have error_increase=None."""
        monitor = DriftMonitor()
        ref = _df({"f": _gauss(seed=0)})
        cur = _df({"f": _gauss(seed=1)})
        report = monitor.detect_data_drift(ref, cur, ["f"])
        assert report.error_increase is None

    # ── Concept drift ─────────────────────────────────────────────────

    def test_concept_drift_detected_when_errors_increase(self) -> None:
        """Large MAE increase should trigger concept drift detection."""
        rng = np.random.default_rng(seed=99)
        baseline_errors = rng.uniform(0, 5, size=200)  # low errors
        recent_errors = rng.uniform(30, 40, size=200)  # high errors
        monitor = DriftMonitor(
            ks_p_threshold=0.05,
            error_increase_threshold=0.20,
        )
        report = monitor.detect_concept_drift(baseline_errors, recent_errors)
        assert report.detected is True
        assert report.recommendation == "retrain"
        assert report.drift_type == "concept"

    def test_no_concept_drift_when_errors_stable(self) -> None:
        """Stable errors from the same distribution must not trigger drift."""
        rng = np.random.default_rng(seed=77)
        baseline_errors = rng.uniform(0, 5, size=300)
        recent_errors = rng.uniform(0, 5, size=300)
        monitor = DriftMonitor(
            ks_p_threshold=0.05,
            error_increase_threshold=0.20,
        )
        report = monitor.detect_concept_drift(baseline_errors, recent_errors)
        assert report.detected is False

    def test_concept_drift_error_increase_field_populated(self) -> None:
        """error_increase must be a float in concept drift report."""
        rng = np.random.default_rng(seed=55)
        baseline = rng.uniform(0, 2, size=100)
        recent = rng.uniform(10, 15, size=100)
        monitor = DriftMonitor()
        report = monitor.detect_concept_drift(baseline, recent)
        assert report.error_increase is not None
        assert isinstance(report.error_increase, float)

    def test_concept_drift_error_increase_positive_when_mae_rises(
        self,
    ) -> None:
        """error_increase must be positive when recent MAE > baseline MAE."""
        baseline = np.ones(100) * 2.0
        recent = np.ones(100) * 5.0  # 150 % increase
        monitor = DriftMonitor()
        report = monitor.detect_concept_drift(baseline, recent)
        assert report.error_increase is not None
        assert report.error_increase > 0.0

    def test_raises_for_empty_baseline_errors(self) -> None:
        """Empty baseline_errors must raise ValueError."""
        monitor = DriftMonitor()
        with pytest.raises(ValueError, match="baseline_errors"):
            monitor.detect_concept_drift(np.array([]), np.ones(50))

    def test_raises_for_empty_recent_errors(self) -> None:
        """Empty recent_errors must raise ValueError."""
        monitor = DriftMonitor()
        with pytest.raises(ValueError, match="recent_errors"):
            monitor.detect_concept_drift(np.ones(50), np.array([]))

    def test_raises_for_non_finite_baseline_errors(self) -> None:
        """Non-finite values in baseline_errors must raise ValueError."""
        monitor = DriftMonitor()
        with pytest.raises(ValueError, match="non-finite"):
            monitor.detect_concept_drift(
                np.array([1.0, float("nan"), 3.0]),
                np.ones(50),
            )

    def test_concept_drift_report_has_empty_affected_features(
        self,
    ) -> None:
        """Concept drift reports must have empty affected_features list."""
        rng = np.random.default_rng(seed=0)
        baseline = rng.uniform(0, 5, size=100)
        recent = rng.uniform(20, 30, size=100)
        monitor = DriftMonitor()
        report = monitor.detect_concept_drift(baseline, recent)
        assert report.affected_features == []

    def test_p_value_key_in_concept_drift_report(self) -> None:
        """Concept drift report must include 'concept_t_test' p-value."""
        rng = np.random.default_rng(seed=1)
        baseline = rng.uniform(0, 5, size=100)
        recent = rng.uniform(0, 5, size=100)
        monitor = DriftMonitor()
        report = monitor.detect_concept_drift(baseline, recent)
        assert "concept_t_test" in report.p_values
