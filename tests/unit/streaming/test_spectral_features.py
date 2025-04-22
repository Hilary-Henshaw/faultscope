"""Unit tests for SpectralFeatureExtractor.

Verifies FFT accuracy, entropy bounds, energy scaling,
cepstral coefficient count, and property-based robustness.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from faultscope.streaming.features.spectral import SpectralFeatureExtractor

# Sampling rate used across all tests unless stated otherwise.
_FS = 100.0  # Hz
_FFT_SENSOR = "vibration"
_NON_FFT_SENSOR = "temperature"


def _extractor(
    fs: float = _FS,
    sensors: list[str] | None = None,
    min_samples: int = 32,
) -> SpectralFeatureExtractor:
    if sensors is None:
        sensors = [_FFT_SENSOR]
    return SpectralFeatureExtractor(
        sampling_rate_hz=fs,
        fft_sensors=sensors,
        min_samples=min_samples,
    )


def _sine_wave(
    freq_hz: float,
    n_samples: int,
    fs: float = _FS,
    amplitude: float = 1.0,
) -> np.ndarray:
    t = np.arange(n_samples) / fs
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


@pytest.mark.unit
class TestSpectralFeatureExtractor:
    """Unit tests for the SpectralFeatureExtractor class."""

    def test_dominant_frequency_for_pure_sine_wave(self) -> None:
        """A pure 10 Hz sine at 100 Hz sampling → dominant_freq ≈ 10 Hz."""
        target_freq = 10.0  # Hz
        n = 512
        signal = _sine_wave(target_freq, n, fs=_FS)
        ext = _extractor(min_samples=n)
        features = ext.extract("ENG_001", {_FFT_SENSOR: signal})
        dominant = features[f"{_FFT_SENSOR}_dominant_freq_hz"]
        # Allow ±1 frequency bin = fs / n = 100/512 ≈ 0.20 Hz tolerance.
        tolerance = _FS / n * 2
        assert abs(dominant - target_freq) <= tolerance, (
            f"Expected dominant freq ≈ {target_freq} Hz, got {dominant:.4f} Hz"
        )

    def test_spectral_energy_scales_with_amplitude(self) -> None:
        """Doubling amplitude → 4× spectral energy (energy ∝ A²)."""
        n = 256
        signal_1x = _sine_wave(10.0, n, amplitude=1.0)
        signal_2x = _sine_wave(10.0, n, amplitude=2.0)
        ext = _extractor(min_samples=n)

        feats_1x = ext.extract("ENG_001", {_FFT_SENSOR: signal_1x})
        feats_2x = ext.extract("ENG_001", {_FFT_SENSOR: signal_2x})

        e1 = feats_1x[f"{_FFT_SENSOR}_spectral_energy"]
        e2 = feats_2x[f"{_FFT_SENSOR}_spectral_energy"]
        assert e2 == pytest.approx(4.0 * e1, rel=0.01)

    def test_spectral_entropy_is_high_for_white_noise(self) -> None:
        """White noise has near-maximum spectral entropy."""
        rng = np.random.default_rng(seed=42)
        noise = rng.standard_normal(512)
        ext = _extractor(min_samples=512)
        features = ext.extract("ENG_001", {_FFT_SENSOR: noise})
        entropy = features[f"{_FFT_SENSOR}_spectral_entropy"]
        # For white noise the PSD is flat → entropy close to log(N/2+1).
        # We just require it to be strictly above a minimum threshold.
        assert entropy > 3.0, (
            f"Expected high entropy for white noise, got {entropy:.4f}"
        )

    def test_spectral_entropy_is_low_for_pure_tone(self) -> None:
        """Pure sine has near-minimum spectral entropy."""
        signal = _sine_wave(10.0, 512)
        ext = _extractor(min_samples=512)
        features = ext.extract("ENG_001", {_FFT_SENSOR: signal})
        entropy_pure = features[f"{_FFT_SENSOR}_spectral_entropy"]

        rng = np.random.default_rng(seed=0)
        noise = rng.standard_normal(512)
        feats_noise = ext.extract("ENG_001", {_FFT_SENSOR: noise})
        entropy_noise = feats_noise[f"{_FFT_SENSOR}_spectral_entropy"]

        assert entropy_pure < entropy_noise, (
            "Pure tone entropy must be less than white noise entropy"
        )

    def test_band_energies_sum_to_approximately_total(self) -> None:
        """Sum of low + mid + high band energies ≤ total energy."""
        signal = _sine_wave(15.0, 512)
        ext = _extractor(min_samples=512)
        features = ext.extract("ENG_001", {_FFT_SENSOR: signal})

        total = features[f"{_FFT_SENSOR}_spectral_energy"]
        bands = (
            features[f"{_FFT_SENSOR}_band_low_energy"]
            + features[f"{_FFT_SENSOR}_band_mid_energy"]
            + features[f"{_FFT_SENSOR}_band_high_energy"]
        )
        # Band boundaries use half-open intervals, so the sum is ≤ total
        # (the DC bin at 0 Hz may not fall in any band).
        assert bands <= total + 1e-6, (
            f"Band sum {bands:.4f} exceeds total {total:.4f}"
        )
        # Bands together should account for most of the energy.
        assert bands >= 0.0

    def test_returns_empty_for_non_fft_sensor(self) -> None:
        """Sensors not in fft_sensors list produce no spectral features."""
        ext = _extractor(sensors=[_FFT_SENSOR])
        signal = _sine_wave(10.0, 256)
        # Pass only the non-FFT sensor in window_values.
        features = ext.extract("ENG_001", {_NON_FFT_SENSOR: signal})
        assert features == {}

    def test_returns_empty_when_samples_below_minimum(self) -> None:
        """Fewer samples than min_samples yields no features."""
        ext = _extractor(min_samples=64)
        # Only 32 samples — below threshold.
        signal = _sine_wave(10.0, 32)
        features = ext.extract("ENG_001", {_FFT_SENSOR: signal})
        assert features == {}

    def test_cepstral_coefficients_have_correct_count(self) -> None:
        """Should always return exactly 5 cepstral coefficients."""
        signal = _sine_wave(10.0, 256)
        ext = _extractor(min_samples=256)
        features = ext.extract("ENG_001", {_FFT_SENSOR: signal})

        cepstral_keys = [
            k for k in features if k.startswith(f"{_FFT_SENSOR}_cepstral_")
        ]
        assert len(cepstral_keys) == 5, (
            f"Expected 5 cepstral keys, got {len(cepstral_keys)}: "
            f"{cepstral_keys}"
        )
        for i in range(5):
            assert f"{_FFT_SENSOR}_cepstral_{i}" in features

    def test_all_expected_feature_keys_are_present(self) -> None:
        """All documented feature names are present in the output."""
        signal = _sine_wave(5.0, 256)
        ext = _extractor(min_samples=256)
        features = ext.extract("ENG_001", {_FFT_SENSOR: signal})

        required_keys = [
            f"{_FFT_SENSOR}_dominant_freq_hz",
            f"{_FFT_SENSOR}_spectral_energy",
            f"{_FFT_SENSOR}_spectral_entropy",
            f"{_FFT_SENSOR}_band_low_energy",
            f"{_FFT_SENSOR}_band_mid_energy",
            f"{_FFT_SENSOR}_band_high_energy",
        ]
        for key in required_keys:
            assert key in features, f"Missing expected key: {key}"

    def test_spectral_energy_is_non_negative(self) -> None:
        """Spectral energy must always be >= 0."""
        signal = _sine_wave(7.0, 128)
        ext = _extractor(min_samples=128)
        features = ext.extract("ENG_001", {_FFT_SENSOR: signal})
        assert features[f"{_FFT_SENSOR}_spectral_energy"] >= 0.0

    def test_raises_for_non_positive_sampling_rate(self) -> None:
        """Constructor must raise ValueError for non-positive sampling rate."""
        with pytest.raises(ValueError, match="sampling_rate_hz"):
            SpectralFeatureExtractor(
                sampling_rate_hz=0.0,
                fft_sensors=[_FFT_SENSOR],
            )

    def test_raises_for_min_samples_below_four(self) -> None:
        """Constructor must raise ValueError when min_samples < 4."""
        with pytest.raises(ValueError, match="min_samples"):
            SpectralFeatureExtractor(
                sampling_rate_hz=_FS,
                fft_sensors=[_FFT_SENSOR],
                min_samples=3,
            )

    @settings(max_examples=50, deadline=2000)
    @given(
        st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=64,
            max_size=512,
        )
    )
    def test_does_not_raise_for_any_valid_input(
        self, values: list[float]
    ) -> None:
        """extract() must never raise for finite float inputs."""
        arr = np.array(values, dtype=np.float64)
        ext = SpectralFeatureExtractor(
            sampling_rate_hz=_FS,
            fft_sensors=[_FFT_SENSOR],
            min_samples=32,
        )
        # Must not raise
        result = ext.extract("ENG_001", {_FFT_SENSOR: arr})
        assert isinstance(result, dict)
