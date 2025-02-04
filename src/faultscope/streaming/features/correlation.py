"""Cross-sensor Pearson correlation feature extraction.

For every configured sensor pair ``(a, b)`` a single feature is
produced::

    {a}_x_{b}_pearson

NaN results (e.g. caused by constant signals) are silently excluded
from the output dictionary.
"""

from __future__ import annotations

import numpy as np

from faultscope.common.logging import get_logger

log = get_logger(__name__)


class CrossSensorCorrelator:
    """Computes Pearson correlation coefficients between sensor pairs.

    Correlation is computed over equal-length value arrays supplied by
    the caller.  Arrays are trimmed to the length of the shorter one
    so that ragged buffers do not cause errors.

    Parameters
    ----------
    sensor_pairs:
        List of ``(sensor_a, sensor_b)`` tuples.  The order within
        each tuple determines the feature name prefix but the
        correlation value itself is symmetric.
    min_samples:
        Minimum number of aligned samples required before a pair is
        processed.  Pairs below this threshold are silently skipped.
    """

    def __init__(
        self,
        sensor_pairs: list[tuple[str, str]],
        min_samples: int = 10,
    ) -> None:
        if min_samples < 2:
            raise ValueError(f"min_samples must be >= 2, got {min_samples}")
        self._pairs: list[tuple[str, str]] = list(sensor_pairs)
        self._min_samples: int = min_samples

    # ── Public API ────────────────────────────────────────────────────

    def extract(
        self,
        window_values: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Return Pearson correlation for every configured pair.

        Parameters
        ----------
        window_values:
            Mapping of sensor name to 1-D float64 value array.  Sensors
            absent from the mapping are skipped.

        Returns
        -------
        dict[str, float]
            Feature name → correlation coefficient.  NaN / missing
            pairs are excluded from the result.
        """
        features: dict[str, float] = {}

        for sensor_a, sensor_b in self._pairs:
            if sensor_a not in window_values or sensor_b not in window_values:
                log.debug(
                    "correlation.missing_sensor",
                    sensor_a=sensor_a,
                    sensor_b=sensor_b,
                )
                continue

            vals_a = window_values[sensor_a]
            vals_b = window_values[sensor_b]

            # Trim to the length of the shorter array so that windows
            # with slightly different depths are still comparable.
            min_len = min(vals_a.shape[0], vals_b.shape[0])
            if min_len < self._min_samples:
                log.debug(
                    "correlation.insufficient_samples",
                    sensor_a=sensor_a,
                    sensor_b=sensor_b,
                    n_samples=min_len,
                    required=self._min_samples,
                )
                continue

            a = vals_a[-min_len:]
            b = vals_b[-min_len:]

            coef = self._pearson(a, b)
            if coef is not None:
                key = f"{sensor_a}_x_{sensor_b}_pearson"
                features[key] = coef

        return features

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
        """Compute the Pearson correlation coefficient.

        Returns ``None`` when either array has zero variance (constant
        signal), which would produce NaN via the standard formula.

        Parameters
        ----------
        a:
            First value array.
        b:
            Second value array (same length as ``a``).

        Returns
        -------
        float | None
            Correlation in ``[-1, 1]``, or ``None`` when undefined.
        """
        std_a = float(np.std(a, ddof=0))
        std_b = float(np.std(b, ddof=0))

        if std_a < 1e-12 or std_b < 1e-12:
            return None

        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        cov = float(np.mean((a - mean_a) * (b - mean_b)))
        return cov / (std_a * std_b)
