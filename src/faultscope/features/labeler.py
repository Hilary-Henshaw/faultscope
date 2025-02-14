"""Training label generators for RUL regression and health classification.

Two labelers are provided:

- ``RulLabeler`` — assigns integer ``rul_cycles`` to each feature row
  based on that machine's position within its observed lifecycle.
- ``HealthLabeler`` — maps ``rul_cycles`` to categorical health status
  strings using configurable thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from faultscope.common.exceptions import ValidationError

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class RulLabeler:
    """Assigns Remaining Useful Life labels to feature rows.

    For each machine's cycle sequence RUL at cycle *t* is defined as::

        RUL(t) = min(total_cycles - t,  max_rul_cycles)

    The label is capped at ``max_rul_cycles`` so the model is not asked
    to predict extremely large values far from any failure event.

    Parameters
    ----------
    max_rul_cycles:
        Hard upper bound applied to all computed RUL values.
    """

    def __init__(self, max_rul_cycles: int = 125) -> None:
        if max_rul_cycles <= 0:
            raise ValidationError(
                "max_rul_cycles must be a positive integer",
                context={"max_rul_cycles": max_rul_cycles},
            )
        self._max_rul = max_rul_cycles

    def assign_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``rul_cycles`` column to *df* in-place and return it.

        The input DataFrame must contain ``machine_id`` and ``cycle``
        columns and be sorted by ``(machine_id, cycle)`` in ascending
        order.  A new integer column ``rul_cycles`` is appended.

        Parameters
        ----------
        df:
            Feature DataFrame with at minimum ``machine_id`` and
            ``cycle`` columns.

        Returns
        -------
        pd.DataFrame
            The same DataFrame with the ``rul_cycles`` column added.

        Raises
        ------
        ValidationError
            If ``machine_id`` or ``cycle`` columns are absent.
        """
        missing = {"machine_id", "cycle"} - set(df.columns)
        if missing:
            raise ValidationError(
                "DataFrame is missing required columns for RUL labelling",
                context={"missing_columns": sorted(missing)},
            )

        df = df.copy()
        df = df.sort_values(["machine_id", "cycle"])

        rul_values: list[int] = []
        for machine_id, group in df.groupby("machine_id", sort=False):
            cycles = group["cycle"].values
            max_cycle = int(cycles.max())
            raw_rul = (max_cycle - cycles).astype(int)
            capped = np.minimum(raw_rul, self._max_rul)
            rul_values.extend(capped.tolist())
            log.debug(
                "rul_assigned",
                machine_id=machine_id,
                max_cycle=max_cycle,
                n_rows=len(group),
            )

        df["rul_cycles"] = rul_values
        log.info(
            "rul_labelling_complete",
            n_rows=len(df),
            rul_min=int(df["rul_cycles"].min()),
            rul_max=int(df["rul_cycles"].max()),
            rul_mean=round(float(df["rul_cycles"].mean()), 2),
        )
        return df


class HealthLabeler:
    """Assigns categorical health status based on RUL thresholds.

    The ``thresholds`` dict maps each label name to the *minimum*
    ``rul_cycles`` value at which that label applies.  Labels are
    evaluated in descending threshold order so the most favourable
    (highest RUL) label wins::

        healthy          → rul_cycles >= healthy_threshold
        degrading        → rul_cycles >= degrading_threshold
        critical         → rul_cycles >= critical_threshold
        imminent_failure → rul_cycles < critical_threshold

    Parameters
    ----------
    thresholds:
        Mapping of ``label_name → lower_bound_rul``.  Expected keys:
        ``"healthy"``, ``"degrading"``, ``"critical"``,
        ``"imminent_failure"``.
    """

    ORDERED_LABELS: list[str] = [
        "healthy",
        "degrading",
        "critical",
        "imminent_failure",
    ]

    def __init__(
        self,
        thresholds: dict[str, int],
    ) -> None:
        missing = set(self.ORDERED_LABELS) - set(thresholds.keys())
        if missing:
            raise ValidationError(
                "thresholds dict is missing required health labels",
                context={"missing_labels": sorted(missing)},
            )
        # Sort descending by threshold value so we can iterate and
        # assign the first matching label.
        self._sorted: list[tuple[str, int]] = sorted(
            thresholds.items(), key=lambda kv: kv[1], reverse=True
        )

    def assign_health(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``health_label`` column derived from ``rul_cycles``.

        Parameters
        ----------
        df:
            DataFrame containing a ``rul_cycles`` integer column.

        Returns
        -------
        pd.DataFrame
            Copy of the input with ``health_label`` appended.

        Raises
        ------
        ValidationError
            If ``rul_cycles`` column is absent.
        """
        if "rul_cycles" not in df.columns:
            raise ValidationError(
                "DataFrame must contain 'rul_cycles' before "
                "health labels can be assigned",
                context={"columns": list(df.columns)},
            )

        df = df.copy()
        rul: pd.Series = df["rul_cycles"]

        # Build a conditions + choices pair for np.select.
        conditions: list[pd.Series] = []
        choices: list[str] = []
        for label, threshold in self._sorted:
            conditions.append(rul >= threshold)
            choices.append(label)

        # np.select evaluates conditions in order; the last label
        # ("imminent_failure" at threshold=0) acts as default because
        # rul >= 0 is always true for non-negative RUL.
        df["health_label"] = np.select(
            conditions, choices, default="imminent_failure"
        )

        counts = df["health_label"].value_counts().to_dict()
        log.info(
            "health_labelling_complete",
            n_rows=len(df),
            label_distribution=counts,
        )
        return df
