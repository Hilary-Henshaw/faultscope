"""Data quality validation and cleaning for incoming sensor readings.

Each ``SensorReading`` is evaluated against a configurable set of
rules.  The outcome is a ``QualityResult`` that carries:

* a ``QualityFlag`` bitmask describing every detected problem,
* a ``filled_readings`` dict with nulls forward-filled from the
  previous reading,
* a boolean ``rejected`` that tells the pipeline whether the message
  should be routed to the DLQ rather than processed.

Rejection conditions:

* ``MISSING_ID``          – ``machine_id`` is empty.
* ``FUTURE_TIMESTAMP``    – ``recorded_at`` is more than
  ``max_future_drift_s`` seconds in the future.
* ``NULL_FRACTION_HIGH``  – more than ``max_null_fraction`` of sensor
  values are NaN/None.

Non-rejection flags (message is cleaned and processed):

* ``NULL_FRACTION_LOW``   – some nulls present but below the threshold;
  forward-filled from the previous reading.
* ``OUTLIER_DETECTED``    – one or more values fall outside
  ``Q1 - 1.5 * IQR`` … ``Q3 + 1.5 * IQR`` (IQR method).
* ``DUPLICATE_TIMESTAMP`` – ``recorded_at`` matches the previous
  reading's timestamp.
* ``SENSOR_COUNT_LOW``    – fewer than ``min_sensor_count`` sensors
  present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Flag, auto

import numpy as np

from faultscope.common.logging import get_logger
from faultscope.streaming.models import SensorReading

log = get_logger(__name__)


class QualityFlag(Flag):
    """Bitmask flags describing data quality issues.

    Multiple flags can be combined:  ``QualityFlag.OUTLIER_DETECTED |
    QualityFlag.NULL_FRACTION_LOW``.
    """

    OK = 0
    MISSING_ID = auto()
    NULL_FRACTION_HIGH = auto()
    NULL_FRACTION_LOW = auto()
    FUTURE_TIMESTAMP = auto()
    DUPLICATE_TIMESTAMP = auto()
    OUTLIER_DETECTED = auto()
    SENSOR_COUNT_LOW = auto()


@dataclass
class QualityResult:
    """Result of a data quality check on one ``SensorReading``.

    Attributes
    ----------
    flags:
        Bitmask of all quality issues detected.
    filled_readings:
        Cleaned sensor values after forward-fill of nulls from the
        previous reading.  Identical to the original readings when no
        nulls were present.
    rejected:
        ``True`` if the message should be routed to the DLQ and not
        processed further.
    """

    flags: QualityFlag
    filled_readings: dict[str, float]
    rejected: bool
    flag_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.flag_names = [
            f.name
            for f in QualityFlag
            if f != QualityFlag.OK and f in self.flags and f.name is not None
        ]


class DataQualityChecker:
    """Validates and cleans incoming ``SensorReading`` instances.

    Parameters
    ----------
    max_null_fraction:
        Fraction of sensors that may be null before the reading is
        rejected to the DLQ.  Readings below this fraction have their
        nulls forward-filled from *previous*.
    max_future_drift_s:
        Maximum number of seconds a reading's timestamp may be ahead
        of wall-clock time before it is rejected.
    min_sensor_count:
        Minimum expected number of sensor keys.  Readings below this
        threshold are flagged (but not rejected).
    """

    def __init__(
        self,
        max_null_fraction: float = 0.3,
        max_future_drift_s: float = 300.0,
        min_sensor_count: int = 5,
    ) -> None:
        if not 0.0 <= max_null_fraction <= 1.0:
            raise ValueError(
                f"max_null_fraction must be in [0, 1], got {max_null_fraction}"
            )
        self._max_null_frac = max_null_fraction
        self._max_future_drift = max_future_drift_s
        self._min_sensor_count = min_sensor_count

    # ── Public API ────────────────────────────────────────────────────

    def check(
        self,
        reading: SensorReading,
        previous: SensorReading | None,
    ) -> QualityResult:
        """Evaluate a reading's quality and return a cleaned version.

        Parameters
        ----------
        reading:
            Incoming sensor reading to validate.
        previous:
            Last accepted reading for the same machine, used for
            forward-fill.  ``None`` on the first reading from a
            machine.

        Returns
        -------
        QualityResult
            Contains the accumulated flags, filled readings, and
            whether the message should be rejected to the DLQ.
        """
        flags = QualityFlag.OK
        reject = False

        # ── 1. Machine ID ─────────────────────────────────────────────
        if not reading.machine_id or not reading.machine_id.strip():
            flags |= QualityFlag.MISSING_ID
            reject = True
            log.warning(
                "quality.missing_id",
                machine_id=repr(reading.machine_id),
            )

        # ── 2. Timestamp drift ────────────────────────────────────────
        now_utc = datetime.now(tz=UTC)
        recorded = reading.recorded_at
        if recorded.tzinfo is None:
            recorded = recorded.replace(tzinfo=UTC)

        drift_s = (recorded - now_utc).total_seconds()
        if drift_s > self._max_future_drift:
            flags |= QualityFlag.FUTURE_TIMESTAMP
            reject = True
            log.warning(
                "quality.future_timestamp",
                machine_id=reading.machine_id,
                drift_s=drift_s,
                limit_s=self._max_future_drift,
            )

        # ── 3. Duplicate timestamp ────────────────────────────────────
        if previous is not None:
            prev_ts = previous.recorded_at
            if prev_ts.tzinfo is None:
                prev_ts = prev_ts.replace(tzinfo=UTC)
            if recorded == prev_ts:
                flags |= QualityFlag.DUPLICATE_TIMESTAMP
                log.debug(
                    "quality.duplicate_timestamp",
                    machine_id=reading.machine_id,
                    recorded_at=recorded.isoformat(),
                )

        # ── 4. Sensor count ───────────────────────────────────────────
        if len(reading.readings) < self._min_sensor_count:
            flags |= QualityFlag.SENSOR_COUNT_LOW
            log.debug(
                "quality.sensor_count_low",
                machine_id=reading.machine_id,
                count=len(reading.readings),
                required=self._min_sensor_count,
            )

        # ── 5. Null fraction ──────────────────────────────────────────
        readings_copy = dict(reading.readings)
        null_count = sum(
            1
            for v in readings_copy.values()
            if v is None or (isinstance(v, float) and math.isnan(v))
        )
        total = max(len(readings_copy), 1)
        null_frac = null_count / total

        if null_count > 0:
            if null_frac > self._max_null_frac:
                flags |= QualityFlag.NULL_FRACTION_HIGH
                reject = True
                log.warning(
                    "quality.null_fraction_high",
                    machine_id=reading.machine_id,
                    null_frac=round(null_frac, 4),
                    threshold=self._max_null_frac,
                )
            else:
                flags |= QualityFlag.NULL_FRACTION_LOW
                readings_copy = self._forward_fill(readings_copy, previous)
                log.debug(
                    "quality.null_filled",
                    machine_id=reading.machine_id,
                    null_count=null_count,
                )

        # ── 6. Outlier detection (IQR) ────────────────────────────────
        if not reject and readings_copy:
            values = np.array(
                [
                    v
                    for v in readings_copy.values()
                    if v is not None
                    and not (isinstance(v, float) and math.isnan(v))
                ],
                dtype=np.float64,
            )
            if values.shape[0] >= 4:
                q1 = float(np.percentile(values, 25))
                q3 = float(np.percentile(values, 75))
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                if np.any(values < lower) or np.any(values > upper):
                    flags |= QualityFlag.OUTLIER_DETECTED
                    log.debug(
                        "quality.outlier_detected",
                        machine_id=reading.machine_id,
                        lower=round(lower, 4),
                        upper=round(upper, 4),
                    )

        return QualityResult(
            flags=flags,
            filled_readings=readings_copy,
            rejected=reject,
        )

    # ── Private helpers ───────────────────────────────────────────────

    @staticmethod
    def _forward_fill(
        readings: dict[str, float],
        previous: SensorReading | None,
    ) -> dict[str, float]:
        """Replace NaN / None values with last-known values.

        Parameters
        ----------
        readings:
            Current sensor readings (may contain NaN / None).
        previous:
            Previous accepted reading used as fill source.

        Returns
        -------
        dict[str, float]
            Readings with nulls replaced.  Sensors with no previous
            value available are replaced with ``0.0``.
        """
        prev_vals: dict[str, float] = (
            previous.readings if previous is not None else {}
        )
        filled: dict[str, float] = {}
        for key, val in readings.items():
            is_null = val is None or (
                isinstance(val, float) and math.isnan(val)
            )
            if is_null:
                filled[key] = prev_vals.get(key, 0.0)
            else:
                filled[key] = val
        return filled
