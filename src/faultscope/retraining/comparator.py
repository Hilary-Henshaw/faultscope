"""A/B model comparison using statistical hypothesis testing.

``ModelComparator`` compares a challenger model against the current
production baseline on a held-out test set.  Promotion is recommended
only when:

1. The improvement is statistically significant (paired t-test,
   p < ``significance``).
2. The challenger is strictly better on all primary metrics.

Usage::

    comparator = ModelComparator(significance=0.05)

    result = comparator.compare_rul_models(
        baseline_predictions=baseline_preds,
        challenger_predictions=challenger_preds,
        ground_truth=rul_labels,
    )
    if result.challenger_better:
        promote_challenger()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

from faultscope.common.logging import get_logger

_log = get_logger(__name__)


@dataclass
class ComparisonResult:
    """Outcome of a head-to-head model comparison.

    Attributes
    ----------
    challenger_better:
        ``True`` when the challenger wins on all metrics and the
        improvement is statistically significant.
    p_value:
        p-value from the paired t-test on per-sample errors.
    delta_mae:
        ``challenger_mae - baseline_mae``.  Negative means challenger
        is better.
    delta_f1:
        ``challenger_f1 - baseline_f1``.  Positive means challenger
        is better.  Zero for RUL comparisons.
    recommendation:
        ``"promote"``, ``"discard"``, or ``"inconclusive"``.
    """

    challenger_better: bool
    p_value: float
    delta_mae: float
    delta_f1: float
    recommendation: str


class ModelComparator:
    """Compares a challenger model against the current production model.

    Uses paired t-tests so that per-sample variance is cancelled out,
    giving more statistical power than an unpaired test.  Both the
    RUL (LSTM, regression) and health-classification (Random Forest)
    comparisons are supported.

    Parameters
    ----------
    significance:
        Alpha level for the paired t-test.  Default 0.05.
    """

    def __init__(self, significance: float = 0.05) -> None:
        self._significance = significance

    def compare_rul_models(
        self,
        baseline_predictions: np.ndarray,
        challenger_predictions: np.ndarray,
        ground_truth: np.ndarray,
    ) -> ComparisonResult:
        """Compare two RUL regressors via paired t-test on absolute errors.

        Lower MAE is better.  The null hypothesis is that challenger
        MAE equals baseline MAE; the alternative is that challenger
        MAE is lower (one-sided).

        Parameters
        ----------
        baseline_predictions:
            RUL point estimates from the current production model.
            Shape ``(n_samples,)``.
        challenger_predictions:
            RUL point estimates from the challenger model.
            Shape ``(n_samples,)``.
        ground_truth:
            Actual RUL values for the test set.
            Shape ``(n_samples,)``.

        Returns
        -------
        ComparisonResult
            Full comparison outcome with recommendation.

        Raises
        ------
        ValueError
            If arrays differ in length or contain fewer than 2 samples.
        """
        self._validate_arrays(
            baseline_predictions,
            challenger_predictions,
            ground_truth,
        )

        baseline_abs_errors = np.abs(baseline_predictions - ground_truth)
        challenger_abs_errors = np.abs(challenger_predictions - ground_truth)

        baseline_mae = float(np.mean(baseline_abs_errors))
        challenger_mae = float(np.mean(challenger_abs_errors))
        delta_mae = challenger_mae - baseline_mae

        # Paired one-sided t-test: H1 = challenger_error < baseline_error
        # i.e. challenger_error - baseline_error < 0.
        diff = challenger_abs_errors - baseline_abs_errors
        t_stat, p_two_sided = stats.ttest_rel(
            challenger_abs_errors,
            baseline_abs_errors,
            alternative="less",
        )
        p_value = float(p_two_sided)

        challenger_better = p_value < self._significance and delta_mae < 0
        recommendation = self._recommend(
            challenger_better=challenger_better,
            p_value=p_value,
        )

        _log.info(
            "model_comparison_rul_complete",
            baseline_mae=round(baseline_mae, 4),
            challenger_mae=round(challenger_mae, 4),
            delta_mae=round(delta_mae, 4),
            t_stat=round(float(t_stat), 4),
            p_value=round(p_value, 6),
            challenger_better=challenger_better,
            recommendation=recommendation,
        )

        _ = diff  # used above to document the sign convention
        return ComparisonResult(
            challenger_better=challenger_better,
            p_value=p_value,
            delta_mae=delta_mae,
            delta_f1=0.0,
            recommendation=recommendation,
        )

    def compare_health_models(
        self,
        baseline_proba: np.ndarray,
        challenger_proba: np.ndarray,
        ground_truth: np.ndarray,
    ) -> ComparisonResult:
        """Compare two health classifiers via macro F1 and critical recall.

        Higher macro F1 is better.  Also checks that the challenger
        does not regress on the ``imminent_failure`` class (index 3),
        since false negatives there are safety-critical.

        Parameters
        ----------
        baseline_proba:
            Softmax probability matrix from baseline.
            Shape ``(n_samples, n_classes)``.
        challenger_proba:
            Softmax probability matrix from challenger.
            Shape ``(n_samples, n_classes)``.
        ground_truth:
            Integer class labels for the test set.
            Shape ``(n_samples,)``.

        Returns
        -------
        ComparisonResult
            Full comparison outcome with recommendation.

        Raises
        ------
        ValueError
            If arrays differ in length or shapes are inconsistent.
        """
        if baseline_proba.shape != challenger_proba.shape:
            raise ValueError(
                "baseline_proba and challenger_proba must have the same"
                f" shape.  Got {baseline_proba.shape} vs "
                f"{challenger_proba.shape}."
            )
        if baseline_proba.shape[0] != ground_truth.shape[0]:
            raise ValueError(
                "Number of probability rows must match ground_truth "
                f"length.  Got {baseline_proba.shape[0]} vs "
                f"{ground_truth.shape[0]}."
            )
        if baseline_proba.shape[0] < 2:
            raise ValueError("At least 2 samples are required for comparison.")

        baseline_preds = np.argmax(baseline_proba, axis=1)
        challenger_preds = np.argmax(challenger_proba, axis=1)

        baseline_f1 = float(
            f1_score(
                ground_truth,
                baseline_preds,
                average="macro",
                zero_division=0,
            )
        )
        challenger_f1 = float(
            f1_score(
                ground_truth,
                challenger_preds,
                average="macro",
                zero_division=0,
            )
        )
        delta_f1 = challenger_f1 - baseline_f1

        # Paired one-sided t-test on per-sample correctness scores.
        # Use probability of correct class as a continuous proxy.
        n_samples = ground_truth.shape[0]
        baseline_correct_proba = baseline_proba[
            np.arange(n_samples), ground_truth
        ]
        challenger_correct_proba = challenger_proba[
            np.arange(n_samples), ground_truth
        ]

        t_stat, p_value = stats.ttest_rel(
            challenger_correct_proba,
            baseline_correct_proba,
            alternative="greater",
        )
        p_value = float(p_value)

        # Safety gate: challenger must not regress on imminent_failure
        # recall if that class exists in the ground truth.
        n_classes = baseline_proba.shape[1]
        critical_class_idx = min(3, n_classes - 1)
        challenger_no_critical_regression = True
        if np.any(ground_truth == critical_class_idx):
            from sklearn.metrics import recall_score

            baseline_recall = float(
                recall_score(
                    ground_truth,
                    baseline_preds,
                    labels=[critical_class_idx],
                    average="macro",
                    zero_division=0,
                )
            )
            challenger_recall = float(
                recall_score(
                    ground_truth,
                    challenger_preds,
                    labels=[critical_class_idx],
                    average="macro",
                    zero_division=0,
                )
            )
            # Allow up to 2 pp regression on critical recall.
            challenger_no_critical_regression = (
                challenger_recall >= baseline_recall - 0.02
            )
            _log.info(
                "model_comparison_health_critical_recall",
                baseline_recall=round(baseline_recall, 4),
                challenger_recall=round(challenger_recall, 4),
                no_regression=challenger_no_critical_regression,
            )

        challenger_better = (
            p_value < self._significance
            and delta_f1 > 0
            and challenger_no_critical_regression
        )
        recommendation = self._recommend(
            challenger_better=challenger_better,
            p_value=p_value,
        )

        baseline_mae_proxy = float(
            np.mean(np.abs(baseline_correct_proba - 1.0))
        )
        challenger_mae_proxy = float(
            np.mean(np.abs(challenger_correct_proba - 1.0))
        )
        delta_mae = challenger_mae_proxy - baseline_mae_proxy

        _log.info(
            "model_comparison_health_complete",
            baseline_f1=round(baseline_f1, 4),
            challenger_f1=round(challenger_f1, 4),
            delta_f1=round(delta_f1, 4),
            t_stat=round(float(t_stat), 4),
            p_value=round(p_value, 6),
            challenger_better=challenger_better,
            recommendation=recommendation,
        )

        return ComparisonResult(
            challenger_better=challenger_better,
            p_value=p_value,
            delta_mae=delta_mae,
            delta_f1=delta_f1,
            recommendation=recommendation,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_arrays(
        baseline: np.ndarray,
        challenger: np.ndarray,
        ground_truth: np.ndarray,
    ) -> None:
        if baseline.shape != challenger.shape:
            raise ValueError(
                "baseline_predictions and challenger_predictions must"
                f" have the same shape.  Got {baseline.shape} vs "
                f"{challenger.shape}."
            )
        if baseline.shape[0] != ground_truth.shape[0]:
            raise ValueError(
                "Predictions and ground_truth must have the same "
                f"length.  Got {baseline.shape[0]} vs "
                f"{ground_truth.shape[0]}."
            )
        if baseline.shape[0] < 2:
            raise ValueError("At least 2 samples are required for comparison.")

    def _recommend(
        self,
        *,
        challenger_better: bool,
        p_value: float,
    ) -> str:
        if challenger_better:
            return "promote"
        if p_value > 0.30:
            return "inconclusive"
        return "discard"
