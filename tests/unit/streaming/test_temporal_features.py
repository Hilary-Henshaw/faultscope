"""Unit tests for TemporalFeatureExtractor.

Tests verify statistical correctness, window isolation,
reset behaviour, and naming convention compliance.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from faultscope.streaming.features.temporal import TemporalFeatureExtractor


def _ts(offset_s: float = 0.0) -> datetime:
    """Return a UTC datetime offset by *offset_s* seconds from epoch."""
    base = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    return base + timedelta(seconds=offset_s)


def _feed(
    extractor: TemporalFeatureExtractor,
    machine_id: str,
    sensor: str,
    values: list[float],
    start_s: float = 0.0,
    step_s: float = 1.0,
) -> None:
    """Push *values* into *extractor* one sample at a time."""
    for i, v in enumerate(values):
        ts = _ts(start_s + i * step_s)
        extractor.update(machine_id, {sensor: v}, ts)


@pytest.mark.unit
class TestTemporalFeatureExtractor:
    """Comprehensive unit tests for TemporalFeatureExtractor."""

    def test_returns_empty_dict_when_insufficient_data(self) -> None:
        """extract() before any update returns empty dict."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        result = ext.extract("ENG_001", _ts(0))
        assert result == {}

    def test_returns_no_windowed_features_after_single_sample(self) -> None:
        """A single update must not produce any windowed (rolling) features."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        ext.update("ENG_001", {"vibration": 1.0}, _ts(0))
        result = ext.extract("ENG_001", _ts(1))
        # Windowed features require >= 2 samples; cumulative features may
        # be present even after the first point.
        windowed = [k for k in result if "_30s_" in k]
        assert windowed == [], f"Unexpected windowed features: {windowed}"

    def test_mean_feature_computed_correctly_for_30s_window(
        self,
    ) -> None:
        """Mean feature equals numpy mean of the same values."""
        ext = TemporalFeatureExtractor(
            window_sizes_s=[30], sampling_rate_hz=1.0
        )
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        _feed(ext, "ENG_001", "vibration", values)
        features = ext.extract("ENG_001", _ts(len(values)))
        key = "vibration_30s_mean"
        assert key in features
        assert abs(features[key] - np.mean(values)) < 1e-9

    def test_std_feature_is_zero_for_constant_signal(self) -> None:
        """Standard deviation must be 0 for a constant input stream."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        _feed(ext, "ENG_001", "temp", [42.0] * 10)
        features = ext.extract("ENG_001", _ts(10))
        assert features["temp_30s_std"] == pytest.approx(0.0, abs=1e-9)

    def test_rms_equals_mean_for_positive_constant(self) -> None:
        """RMS of [c, c, c] equals c when c > 0."""
        c = 5.0
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        _feed(ext, "ENG_001", "pressure", [c] * 8)
        features = ext.extract("ENG_001", _ts(8))
        assert features["pressure_30s_rms"] == pytest.approx(c, rel=1e-6)

    def test_rate_of_change_is_positive_for_increasing_series(
        self,
    ) -> None:
        """Mean absolute diff is positive for a strictly increasing signal."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        _feed(ext, "ENG_001", "temp", [10.0, 20.0, 30.0, 40.0])
        features = ext.extract("ENG_001", _ts(4))
        roc = features["temp_30s_rate_of_change"]
        assert roc > 0.0
        # Each step is +10 → mean abs diff = 10.0
        assert roc == pytest.approx(10.0, rel=1e-6)

    def test_multiple_windows_produce_separate_features(self) -> None:
        """Each window size generates its own set of feature names."""
        ext = TemporalFeatureExtractor(
            window_sizes_s=[30, 60], sampling_rate_hz=1.0
        )
        # Feed 70 seconds of data so both windows accumulate samples.
        _feed(ext, "ENG_001", "vibration", list(range(70)))
        features = ext.extract("ENG_001", _ts(70))
        assert "vibration_30s_mean" in features
        assert "vibration_60s_mean" in features
        # The 60s window has more samples so its mean differs from 30s.
        assert features["vibration_30s_mean"] != features["vibration_60s_mean"]

    def test_reset_clears_window_state_for_machine(self) -> None:
        """After reset(), extract() returns empty dict again."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        _feed(ext, "ENG_001", "vibration", [1.0, 2.0, 3.0, 4.0])
        assert ext.extract("ENG_001", _ts(4)) != {}
        ext.reset("ENG_001")
        assert ext.extract("ENG_001", _ts(5)) == {}

    def test_features_named_with_correct_pattern(self) -> None:
        """Feature names must match {sensor}_{window_s}s_{stat}."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        _feed(ext, "ENG_001", "vibration", [1.0, 2.0, 3.0])
        features = ext.extract("ENG_001", _ts(3))
        expected_stats = {
            "mean",
            "std",
            "min",
            "max",
            "median",
            "rms",
            "range",
            "rate_of_change",
        }
        window_keys = {k for k in features if k.startswith("vibration_30s_")}
        actual_stats = {k.split("vibration_30s_")[1] for k in window_keys}
        assert actual_stats == expected_stats

    def test_handles_nan_values_without_raising(self) -> None:
        """Feeding NaN values must not raise; features remain numeric."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        readings_with_nan = [1.0, float("nan"), 3.0, 4.0]
        for i, v in enumerate(readings_with_nan):
            ext.update("ENG_001", {"sensor": v}, _ts(float(i)))
        # Must not raise
        features = ext.extract("ENG_001", _ts(4))
        # Window has enough non-NaN samples to compute at least some stats
        assert isinstance(features, dict)

    def test_different_machines_have_independent_windows(self) -> None:
        """Windows for distinct machine IDs must not share state."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        _feed(ext, "ENG_001", "vibration", [1.0, 2.0, 3.0, 4.0])
        _feed(ext, "ENG_002", "vibration", [100.0, 200.0, 300.0, 400.0])

        f1 = ext.extract("ENG_001", _ts(4))
        f2 = ext.extract("ENG_002", _ts(4))

        assert f1["vibration_30s_mean"] == pytest.approx(2.5, rel=1e-6)
        assert f2["vibration_30s_mean"] == pytest.approx(250.0, rel=1e-6)

    def test_cumulative_mean_tracks_all_seen_values(self) -> None:
        """Cumulative mean equals the mean of all values ever pushed."""
        ext = TemporalFeatureExtractor(window_sizes_s=[5])
        # Push 20 values with step=1s; window is only 5s wide.
        all_vals = list(range(1, 21))
        _feed(ext, "ENG_001", "sensor", all_vals)
        features = ext.extract("ENG_001", _ts(20))
        assert "sensor_cumulative_mean" in features
        expected = float(np.mean(all_vals))
        assert features["sensor_cumulative_mean"] == pytest.approx(
            expected, rel=1e-6
        )

    def test_cumulative_max_tracks_global_maximum(self) -> None:
        """Cumulative max must equal the largest value pushed so far."""
        ext = TemporalFeatureExtractor(window_sizes_s=[5])
        _feed(ext, "ENG_001", "sensor", [3.0, 1.0, 9.0, 2.0, 4.0])
        features = ext.extract("ENG_001", _ts(5))
        assert features["sensor_cumulative_max"] == pytest.approx(
            9.0, rel=1e-9
        )

    def test_window_range_is_max_minus_min(self) -> None:
        """Range feature must equal max - min of the window values."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        _feed(ext, "ENG_001", "sensor", values)
        features = ext.extract("ENG_001", _ts(5))
        assert features["sensor_30s_range"] == pytest.approx(
            50.0 - 10.0, rel=1e-9
        )

    def test_window_values_returns_correct_arrays(self) -> None:
        """window_values() returns the same data that extract() uses."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        _feed(ext, "ENG_001", "sensor", values)
        wv = ext.window_values("ENG_001", window_s=30)
        assert "sensor" in wv
        assert list(wv["sensor"]) == values

    def test_reset_does_not_affect_other_machines(self) -> None:
        """reset() for one machine must leave all others intact."""
        ext = TemporalFeatureExtractor(window_sizes_s=[30])
        _feed(ext, "ENG_001", "vibration", [1.0, 2.0, 3.0])
        _feed(ext, "ENG_002", "vibration", [4.0, 5.0, 6.0])

        ext.reset("ENG_001")

        assert ext.extract("ENG_001", _ts(4)) == {}
        features_2 = ext.extract("ENG_002", _ts(4))
        assert "vibration_30s_mean" in features_2
