"""Model evaluation metrics for RUL regression and health classification.

``ModelEvaluator`` computes standard regression metrics, the NASA PHM08
asymmetric scoring function, and full classification metrics including
per-class recall for the safety-critical ``imminent_failure`` class.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypedDict

import numpy as np
import structlog
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_HEALTH_LABEL_ORDER: list[str] = [
    "healthy",
    "degrading",
    "critical",
    "imminent_failure",
]


class RulMetrics(TypedDict):
    """Regression metrics for the RUL prediction model."""

    mae: float
    rmse: float
    r2_score: float
    mape: float
    nasa_scoring: float


class HealthMetrics(TypedDict):
    """Classification metrics for the health-status model."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    imminent_failure_recall: float
    confusion_matrix: list[list[int]]


def _nasa_phm_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute the NASA PHM08 asymmetric scoring function.

    Late predictions (underestimating RUL, i.e. *d* < 0) are penalised
    more heavily than early predictions (*d* >= 0) because missing an
    impending failure is more dangerous than a false alarm.

    .. math::

        S = \\sum_{i} s(d_i)

        s(d) = \\exp(-d / 13) - 1  \\text{ if } d < 0

        s(d) = \\exp(d / 10) - 1   \\text{ if } d \\ge 0

    where :math:`d = \\hat{y} - y`.

    A perfect predictor returns ``S = 0``.  Lower (more negative)
    scores indicate systematic under-prediction; large positive scores
    indicate systematic over-prediction.

    Parameters
    ----------
    y_true:
        Ground-truth RUL values.
    y_pred:
        Predicted RUL values.

    Returns
    -------
    float
        Aggregate NASA PHM scoring function value.
    """
    d = y_pred - y_true
    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1.0,
        np.exp(d / 10.0) - 1.0,
    )
    return float(scores.sum())


def _mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-8,
) -> float:
    """Compute MAPE, guarded against near-zero true values.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Predicted values.
    epsilon:
        Small constant added to denominators to avoid division by zero.

    Returns
    -------
    float
        MAPE as a percentage (0–100).
    """
    denom = np.abs(y_true) + epsilon
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


class ModelEvaluator:
    """Computes comprehensive evaluation metrics for both model types.

    Stateless — create one instance and call ``evaluate_rul`` or
    ``evaluate_health`` as needed.
    """

    def evaluate_rul(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> RulMetrics:
        """Compute RUL regression metrics.

        Parameters
        ----------
        y_true:
            Ground-truth RUL values (cycles).
        y_pred:
            Predicted RUL values (cycles).

        Returns
        -------
        RulMetrics
            Dictionary with ``mae``, ``rmse``, ``r2_score``, ``mape``,
            and ``nasa_scoring``.

        Raises
        ------
        ValueError
            If inputs have incompatible shapes.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} "
                f"vs y_pred {y_pred.shape}"
            )

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mape = _mean_absolute_percentage_error(y_true, y_pred)
        nasa = _nasa_phm_score(y_true, y_pred)

        metrics: RulMetrics = {
            "mae": mae,
            "rmse": rmse,
            "r2_score": r2,
            "mape": mape,
            "nasa_scoring": nasa,
        }

        log.info(
            "rul_evaluation_complete",
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            r2=round(r2, 4),
            mape=round(mape, 4),
            nasa=round(nasa, 4),
        )
        return metrics

    def evaluate_health(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray,
    ) -> HealthMetrics:
        """Compute health classification metrics.

        Focuses on recall for the ``imminent_failure`` class because
        false negatives (missing a critical failure) are more dangerous
        than false positives in this domain.

        Parameters
        ----------
        y_true:
            Ground-truth health label strings, shape ``(n_samples,)``.
        y_pred:
            Predicted health label strings, shape ``(n_samples,)``.
        proba:
            Class probability matrix, shape
            ``(n_samples, n_classes)``.  Columns must align with
            ``_HEALTH_LABEL_ORDER``.

        Returns
        -------
        HealthMetrics
            Dictionary with ``accuracy``, ``macro_f1``,
            ``weighted_f1``, ``imminent_failure_recall``, and
            ``confusion_matrix``.
        """
        acc = float(accuracy_score(y_true, y_pred))

        macro_f1 = float(
            f1_score(
                y_true,
                y_pred,
                labels=_HEALTH_LABEL_ORDER,
                average="macro",
                zero_division=0,
            )
        )
        weighted_f1 = float(
            f1_score(
                y_true,
                y_pred,
                labels=_HEALTH_LABEL_ORDER,
                average="weighted",
                zero_division=0,
            )
        )

        # Per-class recall for imminent_failure specifically.
        per_class_recall: np.ndarray = recall_score(
            y_true,
            y_pred,
            labels=_HEALTH_LABEL_ORDER,
            average=None,  # type: ignore[arg-type]
            zero_division=0,
        )
        imm_idx = _HEALTH_LABEL_ORDER.index("imminent_failure")
        imm_recall = float(per_class_recall[imm_idx])

        cm: np.ndarray = confusion_matrix(
            y_true,
            y_pred,
            labels=_HEALTH_LABEL_ORDER,
        )

        metrics: HealthMetrics = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "imminent_failure_recall": imm_recall,
            "confusion_matrix": cm.tolist(),
        }

        log.info(
            "health_evaluation_complete",
            accuracy=round(acc, 4),
            macro_f1=round(macro_f1, 4),
            weighted_f1=round(weighted_f1, 4),
            imminent_failure_recall=round(imm_recall, 4),
        )
        return metrics

    def generate_model_card(
        self,
        model_type: str,
        metrics: RulMetrics | HealthMetrics,
        feature_names: list[str],
        training_data_info: dict[str, object],
    ) -> dict[str, object]:
        """Generate a model card dictionary for documentation.

        The model card follows the Google Model Card format and
        includes model identity, intended use, evaluation results,
        training data summary, and limitations.

        Parameters
        ----------
        model_type:
            ``"lifespan_predictor"`` or ``"condition_classifier"``.
        metrics:
            Evaluation metrics from ``evaluate_rul`` or
            ``evaluate_health``.
        feature_names:
            Ordered list of feature column names used during training.
        training_data_info:
            Arbitrary metadata about the training dataset, e.g.
            ``{"n_rows": 100000, "dataset_version": "v1"}``.

        Returns
        -------
        dict[str, object]
            Model card as a nested dictionary suitable for JSON
            serialisation.
        """
        intended_use: dict[str, object]
        if model_type == "lifespan_predictor":
            intended_use = {
                "primary_use": (
                    "Predict remaining useful life (in cycles) "
                    "for manufacturing equipment."
                ),
                "out_of_scope": (
                    "Not suitable for safety-critical decisions "
                    "without human review."
                ),
            }
        else:
            intended_use = {
                "primary_use": (
                    "Classify equipment health status into "
                    "four ordinal categories for maintenance scheduling."
                ),
                "out_of_scope": (
                    "Not a substitute for physical sensor diagnostics."
                ),
            }

        card: dict[str, object] = {
            "model_details": {
                "name": model_type,
                "framework": (
                    "TensorFlow/Keras"
                    if model_type == "lifespan_predictor"
                    else "scikit-learn RandomForest"
                ),
                "generated_at": datetime.now(tz=UTC).isoformat(),
                "version": training_data_info.get("dataset_version", "v1"),
            },
            "intended_use": intended_use,
            "training_data": {
                **training_data_info,
                "n_features": len(feature_names),
                "feature_names": feature_names[:50],
            },
            "evaluation_results": {
                "metrics": dict(metrics),
                "evaluation_dataset": "held-out test split",
            },
            "limitations": [
                "Model performance degrades on machine types not "
                "present in training data.",
                "RUL predictions assume similar operational conditions "
                "to the training distribution.",
                "Prediction intervals are estimated via MC Dropout and "
                "may undercover at the tails.",
            ],
            "ethical_considerations": [
                "Automated maintenance scheduling should always be "
                "reviewed by a qualified engineer.",
                "False-negative imminent-failure predictions can "
                "result in unsafe conditions.",
            ],
        }
        return card
