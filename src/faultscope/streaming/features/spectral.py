"""Frequency-domain feature extraction via FFT.

For each sensor listed in ``fft_sensors`` and for which sufficient
buffered samples are available, the following features are produced:

* ``{sensor}_dominant_freq_hz`` - peak frequency in the power spectrum
* ``{sensor}_spectral_energy``  - sum of power spectral density
* ``{sensor}_spectral_entropy`` - Shannon entropy of normalised PSD
* ``{sensor}_band_low_energy``  - energy in 0 – 0.1 * Nyquist
* ``{sensor}_band_mid_energy``  - energy in 0.1 – 0.3 * Nyquist
* ``{sensor}_band_high_energy`` - energy in 0.3 – 0.5 * Nyquist (=
  Nyquist itself)
* ``{sensor}_cepstral_0`` … ``{sensor}_cepstral_4`` – first five real
  cepstral coefficients
"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

from faultscope.common.logging import get_logger

log = get_logger(__name__)

# Band boundaries as fraction of Nyquist frequency.
_BAND_LOW_HI: float = 0.1
_BAND_MID_HI: float = 0.3
_BAND_HIGH_HI: float = 0.5  # == Nyquist

# Tiny constant to avoid log(0) in entropy computation.
_EPS: float = 1e-12

# Number of cepstral coefficients to keep.
_N_CEPSTRAL: int = 5


class SpectralFeatureExtractor:
    """Computes frequency-domain features via real FFT.

    Parameters
    ----------
    sampling_rate_hz:
        Sample rate assumed for all sensors (used to compute frequency
        axis via ``rfftfreq``).
    fft_sensors:
        Whitelist of sensor names for which spectral features are
        computed.  Sensors absent from this list are silently skipped.
    min_samples:
        Minimum number of buffered samples required before computation
        is attempted.  Must be at least 4 to produce meaningful FFT
        output; defaults to 32.
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        fft_sensors: list[str],
        min_samples: int = 32,
    ) -> None:
        if sampling_rate_hz <= 0:
            raise ValueError(
                f"sampling_rate_hz must be positive, got {sampling_rate_hz}"
            )
        if min_samples < 4:
            raise ValueError(f"min_samples must be >= 4, got {min_samples}")
        self._fs: float = sampling_rate_hz
        self._sensors: frozenset[str] = frozenset(fft_sensors)
        self._min_samples: int = min_samples
        self._nyquist: float = sampling_rate_hz / 2.0

    # ── Public API ────────────────────────────────────────────────────

    def extract(
        self,
        machine_id: str,
        window_values: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Compute spectral features from buffered window arrays.

        Parameters
        ----------
        machine_id:
            Identifier of the originating machine (used for logging).
        window_values:
            Mapping of sensor name to 1-D float64 array of buffered
            values produced by ``TemporalFeatureExtractor.window_values``.

        Returns
        -------
        dict[str, float]
            Spectral feature dict.  Returns an empty dict when no
            sensor passes the sensor-whitelist and min-samples checks.
        """
        features: dict[str, float] = {}

        for sensor, vals in window_values.items():
            if sensor not in self._sensors:
                continue
            if vals.shape[0] < self._min_samples:
                log.debug(
                    "spectral.insufficient_samples",
                    machine_id=machine_id,
                    sensor=sensor,
                    n_samples=vals.shape[0],
                    required=self._min_samples,
                )
                continue

            sensor_feats = self._compute_sensor_features(sensor, vals)
            features.update(sensor_feats)

        return features

    # ── Private helpers ───────────────────────────────────────────────

    def _compute_sensor_features(
        self, sensor: str, vals: np.ndarray
    ) -> dict[str, float]:
        """Run the full spectral pipeline for a single sensor."""
        n = vals.shape[0]

        # Remove DC offset before FFT to improve frequency resolution.
        x = vals - np.mean(vals)

        # Real FFT → magnitude spectrum → power spectral density.
        spectrum: np.ndarray = rfft(x, n=n)
        freqs: np.ndarray = rfftfreq(n, d=1.0 / self._fs)
        psd: np.ndarray = np.abs(spectrum) ** 2

        total_energy = float(np.sum(psd))
        dominant_idx = int(np.argmax(psd))
        dominant_freq = float(freqs[dominant_idx])

        # Normalised PSD for entropy computation.
        if total_energy > _EPS:
            p_norm = psd / total_energy
        else:
            p_norm = np.ones_like(psd) / max(len(psd), 1)

        entropy = float(-np.sum(p_norm * np.log(p_norm + _EPS)))

        # Band energies.
        band_low = self._band_energy(
            psd, freqs, 0.0, _BAND_LOW_HI * self._nyquist
        )
        band_mid = self._band_energy(
            psd,
            freqs,
            _BAND_LOW_HI * self._nyquist,
            _BAND_MID_HI * self._nyquist,
        )
        band_high = self._band_energy(
            psd,
            freqs,
            _BAND_MID_HI * self._nyquist,
            _BAND_HIGH_HI * self._nyquist,
        )

        # Real cepstrum: IFFT(log(|FFT(x)|)).
        log_mag = np.log(np.abs(spectrum) + _EPS)
        # irfft expects a one-sided spectrum; produce full cepstrum.
        cepstrum: np.ndarray = np.real(irfft(log_mag, n=n))
        cepstral_coefs = cepstrum[:_N_CEPSTRAL]

        result: dict[str, float] = {
            f"{sensor}_dominant_freq_hz": dominant_freq,
            f"{sensor}_spectral_energy": total_energy,
            f"{sensor}_spectral_entropy": entropy,
            f"{sensor}_band_low_energy": band_low,
            f"{sensor}_band_mid_energy": band_mid,
            f"{sensor}_band_high_energy": band_high,
        }

        for i, coef in enumerate(cepstral_coefs):
            result[f"{sensor}_cepstral_{i}"] = float(coef)

        # Pad missing cepstral coefficients with 0.0 when the signal
        # was shorter than _N_CEPSTRAL samples.
        for i in range(len(cepstral_coefs), _N_CEPSTRAL):
            result[f"{sensor}_cepstral_{i}"] = 0.0

        return result

    @staticmethod
    def _band_energy(
        psd: np.ndarray,
        freqs: np.ndarray,
        low_hz: float,
        high_hz: float,
    ) -> float:
        """Sum PSD values whose corresponding frequency falls in
        ``[low_hz, high_hz)``.

        Parameters
        ----------
        psd:
            Power spectral density array (length = ``n // 2 + 1``).
        freqs:
            Frequency axis produced by ``rfftfreq``.
        low_hz:
            Lower bound of the frequency band (inclusive).
        high_hz:
            Upper bound of the frequency band (exclusive).

        Returns
        -------
        float
            Total energy in the band.
        """
        mask = (freqs >= low_hz) & (freqs < high_hz)
        return float(np.sum(psd[mask]))
