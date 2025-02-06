"""Time-domain feature extraction for sensor readings.

Maintains per-machine, per-sensor sliding windows and computes a
comprehensive set of statistical features each time ``extract`` is
called.

Feature naming convention::

    {sensor}_{window_s}s_{stat}

Cumulative statistics (across all data seen since last reset)::

    {sensor}_cumulative_mean
    {sensor}_cumulative_max
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

import numpy as np

from faultscope.common.logging import get_logger

log = get_logger(__name__)

# Number of values retained for rate-of-change computation.
_ROC_MIN_SAMPLES: int = 2


class SensorWindow:
    """Fixed-duration sliding window for a single sensor channel.

    Internally stores ``(timestamp, value)`` pairs and drops entries
    that fall outside the ``[now - window_s, now]`` interval on each
    ``push``.

    Parameters
    ----------
    window_s:
        Duration of the window in seconds.
    sampling_rate_hz:
        Expected sensor sampling rate.  Used only to pre-size the
        internal deque so memory is bounded at initialisation time.
    """

    def __init__(
        self,
        window_s: int,
        sampling_rate_hz: float = 1.0,
    ) -> None:
        self._window_s: int = window_s
        max_len = max(2, int(window_s * sampling_rate_hz * 2))
        self._buf: deque[tuple[datetime, float]] = deque(maxlen=max_len)

    def push(self, value: float, timestamp: datetime) -> None:
        """Append a new sample and evict samples outside the window.

        Parameters
        ----------
        value:
            Sensor reading value.
        timestamp:
            UTC timestamp of the sample.
        """
        self._buf.append((timestamp, value))
        cutoff = timestamp.timestamp() - self._window_s
        while self._buf and self._buf[0][0].timestamp() < cutoff:
            self._buf.popleft()

    def is_ready(self) -> bool:
        """Return ``True`` when the window contains at least 2 samples."""
        return len(self._buf) >= _ROC_MIN_SAMPLES

    def values(self) -> np.ndarray:
        """Return the buffered values as a 1-D float64 array."""
        return np.array([v for _, v in self._buf], dtype=np.float64)


class TemporalFeatureExtractor:
    """Computes rolling statistical features over configurable windows.

    Maintains independent ``SensorWindow`` instances per
    ``(machine_id, sensor, window_s)`` triple and separate cumulative
    accumulators per ``(machine_id, sensor)``.

    For each sensor and window size the following features are produced:

    * ``{sensor}_{window_s}s_mean``
    * ``{sensor}_{window_s}s_std``
    * ``{sensor}_{window_s}s_min``
    * ``{sensor}_{window_s}s_max``
    * ``{sensor}_{window_s}s_median``
    * ``{sensor}_{window_s}s_rms``
    * ``{sensor}_{window_s}s_range``
    * ``{sensor}_{window_s}s_rate_of_change``

    Additionally, cumulative (session-level) statistics:

    * ``{sensor}_cumulative_mean``
    * ``{sensor}_cumulative_max``

    Parameters
    ----------
    window_sizes_s:
        List of window durations in seconds for which features are
        computed.
    sampling_rate_hz:
        Expected sensor sampling rate used to size internal buffers.
    """

    def __init__(
        self,
        window_sizes_s: list[int],
        sampling_rate_hz: float = 1.0,
    ) -> None:
        self._window_sizes: list[int] = sorted(window_sizes_s)
        self._sampling_rate: float = sampling_rate_hz

        # {machine_id: {sensor: {window_s: SensorWindow}}}
        self._windows: dict[str, dict[str, dict[int, SensorWindow]]] = {}

        # Welford online accumulators for cumulative stats
        # {machine_id: {sensor: (count, mean, M2, running_max)}}
        self._cumulative: dict[
            str,
            dict[str, tuple[int, float, float, float]],
        ] = {}

    # ── Public API ────────────────────────────────────────────────────

    def update(
        self,
        machine_id: str,
        readings: dict[str, float],
        timestamp: datetime,
    ) -> None:
        """Feed new sensor readings into all internal windows.

        Parameters
        ----------
        machine_id:
            Identifier of the originating machine.
        readings:
            Mapping of sensor name to measured value.
        timestamp:
            UTC timestamp of the readings batch.
        """
        self._ensure_machine(machine_id, list(readings.keys()))

        for sensor, value in readings.items():
            for window_s in self._window_sizes:
                self._windows[machine_id][sensor][window_s].push(
                    value, timestamp
                )
            self._update_cumulative(machine_id, sensor, value)

    def extract(
        self,
        machine_id: str,
        timestamp: datetime,
    ) -> dict[str, float]:
        """Return all temporal features for *machine_id*.

        Returns an empty dict when the machine has no data or when no
        window has accumulated sufficient samples.

        Parameters
        ----------
        machine_id:
            Target machine identifier.
        timestamp:
            Current processing time (used for logging only).
        """
        if machine_id not in self._windows:
            log.debug(
                "temporal.no_state",
                machine_id=machine_id,
                ts=timestamp.isoformat(),
            )
            return {}

        features: dict[str, float] = {}

        for sensor, window_map in self._windows[machine_id].items():
            for window_s, win in window_map.items():
                if not win.is_ready():
                    continue
                vals = win.values()
                prefix = f"{sensor}_{window_s}s"
                features.update(self._compute_window_stats(prefix, vals))

            # Cumulative stats
            if sensor in self._cumulative[machine_id]:
                count, mean, _, running_max = self._cumulative[machine_id][
                    sensor
                ]
                if count > 0:
                    features[f"{sensor}_cumulative_mean"] = mean
                    features[f"{sensor}_cumulative_max"] = running_max

        return features

    def reset(self, machine_id: str) -> None:
        """Clear all window and cumulative state for *machine_id*.

        Parameters
        ----------
        machine_id:
            Machine whose state should be cleared (e.g., after a
            detected restart event).
        """
        self._windows.pop(machine_id, None)
        self._cumulative.pop(machine_id, None)
        log.info("temporal.reset", machine_id=machine_id)

    def window_values(
        self,
        machine_id: str,
        window_s: int,
    ) -> dict[str, np.ndarray]:
        """Return raw window arrays for a machine at the given size.

        Used by ``SpectralFeatureExtractor`` to obtain the same buffer
        contents without duplication.

        Parameters
        ----------
        machine_id:
            Target machine.
        window_s:
            Window size in seconds.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping of sensor name to buffered value array.  Sensors
            with fewer than 2 samples are omitted.
        """
        if machine_id not in self._windows:
            return {}
        result: dict[str, np.ndarray] = {}
        for sensor, window_map in self._windows[machine_id].items():
            if window_s not in window_map:
                continue
            win = window_map[window_s]
            if win.is_ready():
                result[sensor] = win.values()
        return result

    # ── Private helpers ───────────────────────────────────────────────

    def _ensure_machine(self, machine_id: str, sensors: list[str]) -> None:
        """Initialise per-machine state if it does not exist."""
        if machine_id not in self._windows:
            self._windows[machine_id] = {}
            self._cumulative[machine_id] = {}

        for sensor in sensors:
            if sensor not in self._windows[machine_id]:
                self._windows[machine_id][sensor] = {
                    w: SensorWindow(w, self._sampling_rate)
                    for w in self._window_sizes
                }
            if sensor not in self._cumulative[machine_id]:
                # (count, mean, M2, running_max)
                self._cumulative[machine_id][sensor] = (
                    0,
                    0.0,
                    0.0,
                    float("-inf"),
                )

    def _update_cumulative(
        self, machine_id: str, sensor: str, value: float
    ) -> None:
        """Update Welford online mean/variance accumulator."""
        count, mean, m2, running_max = self._cumulative[machine_id][sensor]
        count += 1
        delta = value - mean
        mean += delta / count
        delta2 = value - mean
        m2 += delta * delta2
        running_max = max(running_max, value)
        self._cumulative[machine_id][sensor] = (
            count,
            mean,
            m2,
            running_max,
        )

    @staticmethod
    def _compute_window_stats(
        prefix: str, vals: np.ndarray
    ) -> dict[str, float]:
        """Compute all per-window statistics for a value array.

        Parameters
        ----------
        prefix:
            Feature name prefix, e.g. ``"vibration_rms_30s"``.
        vals:
            Non-empty 1-D float64 array of sensor readings.

        Returns
        -------
        dict[str, float]
            Computed statistics keyed by full feature name.
        """
        mean: float = float(np.mean(vals))
        std: float = float(np.std(vals, ddof=0))
        vmin: float = float(np.min(vals))
        vmax: float = float(np.max(vals))
        median: float = float(np.median(vals))
        rms: float = float(np.sqrt(np.mean(vals**2)))
        vrange: float = vmax - vmin

        if vals.shape[0] >= _ROC_MIN_SAMPLES:
            diffs = np.diff(vals)
            roc: float = float(np.mean(np.abs(diffs)))
        else:
            roc = 0.0

        return {
            f"{prefix}_mean": mean,
            f"{prefix}_std": std,
            f"{prefix}_min": vmin,
            f"{prefix}_max": vmax,
            f"{prefix}_median": median,
            f"{prefix}_rms": rms,
            f"{prefix}_range": vrange,
            f"{prefix}_rate_of_change": roc,
        }
