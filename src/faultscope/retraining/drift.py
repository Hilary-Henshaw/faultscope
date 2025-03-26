"""Statistical drift detection for the FaultScope retraining pipeline.

Provides ``DriftMonitor`` which detects two categories of drift:

* **Data drift** (covariate shift): the marginal distribution of input
  features has changed.  Detected via a two-sample Kolmogorov-Smirnov
  test on each feature independently.
* **Concept drift**: the relationship between features and the target
  has changed, manifesting as a significant increase in prediction
  error over time.  Detected via a one-sided t-test on mean absolute
  errors.

Usage::

    monitor = DriftMonitor(ks_p_threshold=0.05)
    report = monitor.detect_data_drift(
        reference_df=ref,
        current_df=cur,
        feature_cols=["vibration_x", "temperature"],
    )
    if report.detected:
        print(report.recommendation)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from faultscope.common.logging import get_logger

_log = get_logger(__name__)


@dataclass
class DriftReport:
    """Summary of a single drift-detection run.

    Attributes
    ----------
    detected:
        ``True`` when at least one drift signal was found.
    drift_type:
        Category of drift: ``"data"``, ``"concept"``, or
        ``"covariate"``.
    affected_features:
        Names of features whose distribution shifted significantly.
        Empty for concept-drift reports.
    ks_statistics:
        KS test statistic per feature (only for data/covariate drift).
    p_values:
        KS test p-value per feature (only for data/covariate drift).
    error_increase:
        Fractional increase in mean absolute error vs baseline.
        ``None`` for data/covariate drift reports.
    recommendation:
        High-level action: ``"retrain"``, ``"monitor"``, or ``"ok"``.
    """

    detected: bool
    drift_type: str
    affected_features: list[str] = field(default_factory=list)
    ks_statistics: dict[str, float] = field(default_factory=dict)
    p_values: dict[str, float] = field(default_factory=dict)
    error_increase: float | None = None
    recommendation: str = "ok"


class DriftMonitor:
    """Detects data drift and concept drift in model inputs/outputs.

    Data drift:
        Two-sample KS test on each feature column comparing a
        reference window (training distribution) against the current
        window (live production data).  Features whose p-value falls
        below ``ks_p_threshold`` are flagged.

    Concept drift:
        One-sided independent t-test comparing mean absolute errors
        from the recent production window against a stable baseline
        window.  Drift is declared when the recent MAE is significantly
        higher *and* the fractional increase exceeds
        ``error_increase_threshold``.

    Parameters
    ----------
    ks_p_threshold:
        KS test p-value threshold.  Features with
        ``p < ks_p_threshold`` are considered drifted.
    error_increase_threshold:
        Minimum fractional MAE increase to trigger concept-drift
        detection.  Default 0.20 means 20% increase.
    """

    def __init__(
        self,
        ks_p_threshold: float = 0.05,
        error_increase_threshold: float = 0.20,
    ) -> None:
        self._ks_p_threshold = ks_p_threshold
        self._error_increase_threshold = error_increase_threshold

    def detect_data_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> DriftReport:
        """Run a KS test on each feature and aggregate results.

        For each feature in ``feature_cols``, runs a two-sample KS
        test between the reference and current distributions.  A
        feature is flagged when its p-value is below
        ``ks_p_threshold``.

        Parameters
        ----------
        reference_df:
            Historical feature DataFrame representing the training
            distribution.
        current_df:
            Recent production feature DataFrame to compare against.
        feature_cols:
            Column names to test.  Both DataFrames must contain these
            columns.

        Returns
        -------
        DriftReport
            Report with per-feature KS statistics, p-values, and an
            overall recommendation.

        Raises
        ------
        ValueError
            If a required column is missing from either DataFrame or
            if the DataFrames are empty.
        """
        missing_ref = set(feature_cols) - set(reference_df.columns)
        missing_cur = set(feature_cols) - set(current_df.columns)
        if missing_ref:
            raise ValueError(
                f"Reference DataFrame missing columns: {missing_ref}"
            )
        if missing_cur:
            raise ValueError(
                f"Current DataFrame missing columns: {missing_cur}"
            )
        if reference_df.empty:
            raise ValueError("reference_df must not be empty.")
        if current_df.empty:
            raise ValueError("current_df must not be empty.")

        ks_stats: dict[str, float] = {}
        p_values: dict[str, float] = {}
        drifted: list[str] = []

        for col in feature_cols:
            ref_vals = reference_df[col].dropna().to_numpy()
            cur_vals = current_df[col].dropna().to_numpy()

            if ref_vals.size == 0 or cur_vals.size == 0:
                _log.warning(
                    "drift_ks_empty_column",
                    feature=col,
                    ref_rows=ref_vals.size,
                    cur_rows=cur_vals.size,
                )
                continue

            result = stats.ks_2samp(ref_vals, cur_vals)
            ks_stats[col] = float(result.statistic)
            p_values[col] = float(result.pvalue)

            if result.pvalue < self._ks_p_threshold:
                drifted.append(col)
                _log.info(
                    "drift_ks_feature_flagged",
                    feature=col,
                    ks_stat=round(result.statistic, 4),
                    p_value=round(result.pvalue, 6),
                    threshold=self._ks_p_threshold,
                )

        detected = len(drifted) > 0
        if detected:
            recommendation = "retrain"
        elif any(p < self._ks_p_threshold * 3 for p in p_values.values()):
            recommendation = "monitor"
        else:
            recommendation = "ok"

        _log.info(
            "drift_data_detection_complete",
            detected=detected,
            n_drifted=len(drifted),
            n_tested=len(feature_cols),
            recommendation=recommendation,
        )

        return DriftReport(
            detected=detected,
            drift_type="covariate",
            affected_features=drifted,
            ks_statistics=ks_stats,
            p_values=p_values,
            error_increase=None,
            recommendation=recommendation,
        )

    def detect_concept_drift(
        self,
        baseline_errors: np.ndarray,
        recent_errors: np.ndarray,
    ) -> DriftReport:
        """Check whether recent prediction errors have significantly increased.

        Uses a one-sided Welch's t-test (unequal variances) to determine
        whether the mean of ``recent_errors`` is significantly higher
        than the mean of ``baseline_errors``.  The test is combined
        with a fractional-increase check: both conditions must hold.

        Parameters
        ----------
        baseline_errors:
            Array of absolute prediction errors from the stable
            production baseline window.
        recent_errors:
            Array of absolute prediction errors from the recent
            production window.

        Returns
        -------
        DriftReport
            Report with ``error_increase``, ``drift_type="concept"``,
            and a recommendation.

        Raises
        ------
        ValueError
            If either error array is empty or contains non-finite values.
        """
        if baseline_errors.size == 0:
            raise ValueError("baseline_errors must not be empty.")
        if recent_errors.size == 0:
            raise ValueError("recent_errors must not be empty.")
        if not np.all(np.isfinite(baseline_errors)):
            raise ValueError("baseline_errors contains non-finite values.")
        if not np.all(np.isfinite(recent_errors)):
            raise ValueError("recent_errors contains non-finite values.")

        baseline_mae = float(np.mean(np.abs(baseline_errors)))
        recent_mae = float(np.mean(np.abs(recent_errors)))

        if baseline_mae == 0.0:
            error_increase = 0.0
        else:
            error_increase = (recent_mae - baseline_mae) / baseline_mae

        # One-sided Welch's t-test: H1 = recent_mae > baseline_mae.
        t_stat, p_two_sided = stats.ttest_ind(
            np.abs(recent_errors),
            np.abs(baseline_errors),
            equal_var=False,
            alternative="greater",
        )
        # scipy returns the one-sided p-value when alternative="greater".
        p_one_sided = float(p_two_sided)

        statistically_significant = p_one_sided < self._ks_p_threshold
        magnitude_significant = error_increase > self._error_increase_threshold
        detected = statistically_significant and magnitude_significant

        if detected:
            recommendation = "retrain"
        elif error_increase > self._error_increase_threshold * 0.5:
            recommendation = "monitor"
        else:
            recommendation = "ok"

        _log.info(
            "drift_concept_detection_complete",
            detected=detected,
            baseline_mae=round(baseline_mae, 4),
            recent_mae=round(recent_mae, 4),
            error_increase=round(error_increase, 4),
            t_stat=round(float(t_stat), 4),
            p_one_sided=round(p_one_sided, 6),
            recommendation=recommendation,
        )

        return DriftReport(
            detected=detected,
            drift_type="concept",
            affected_features=[],
            ks_statistics={},
            p_values={"concept_t_test": p_one_sided},
            error_increase=error_increase,
            recommendation=recommendation,
        )
