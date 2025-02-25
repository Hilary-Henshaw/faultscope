"""Time-series aware cross-validation for faultscope models.

Walk-forward (expanding window) splits ensure that test data always
comes strictly after the corresponding training data so that no
temporal leakage occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union, cast

import numpy as np
import pandas as pd
import structlog

from faultscope.common.exceptions import ValidationError

if TYPE_CHECKING:
    from faultscope.training.models.condition_classifier import (
        ConditionClassifier,
    )
    from faultscope.training.models.lifespan_predictor import (
        LifespanPredictor,
    )

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

AnyModel = Union["LifespanPredictor", "ConditionClassifier"]


@dataclass
class TimeSeriesFold:
    """Container for a single train/test fold.

    Attributes
    ----------
    fold_id:
        Zero-based fold index.
    train_indices:
        Integer positions of training rows in the original DataFrame.
    test_indices:
        Integer positions of test rows in the original DataFrame.
    """

    fold_id: int
    train_indices: np.ndarray = field(repr=False)
    test_indices: np.ndarray = field(repr=False)


class TimeSeriesCrossValidator:
    """Walk-forward time-series cross-validation.

    Splits a DataFrame respecting temporal ordering — the test window
    always follows the training window.  The initial training window
    is sized so that there are exactly ``n_folds`` non-overlapping
    test windows of equal size covering the tail of the dataset.

    An optional *gap_cycles* parameter drops a buffer of rows between
    each training and test window to avoid look-ahead bias when
    features are computed over rolling windows.

    Parameters
    ----------
    n_folds:
        Number of cross-validation folds.
    gap_cycles:
        Number of rows to skip between the end of training and the
        start of testing in each fold.
    """

    def __init__(
        self,
        n_folds: int = 5,
        gap_cycles: int = 0,
    ) -> None:
        if n_folds < 2:
            raise ValidationError(
                "n_folds must be at least 2",
                context={"n_folds": n_folds},
            )
        if gap_cycles < 0:
            raise ValidationError(
                "gap_cycles must be non-negative",
                context={"gap_cycles": gap_cycles},
            )
        self._n_folds = n_folds
        self._gap = gap_cycles

    def split(
        self,
        df: pd.DataFrame,
        time_col: str = "computed_at",
    ) -> list[TimeSeriesFold]:
        """Generate walk-forward train/test index splits.

        Parameters
        ----------
        df:
            DataFrame to split, must contain *time_col*.
        time_col:
            Name of the datetime column used to determine temporal
            ordering.

        Returns
        -------
        list[TimeSeriesFold]
            One ``TimeSeriesFold`` per fold, in chronological order.

        Raises
        ------
        ValidationError
            If *time_col* is absent or the dataset is too small.
        """
        if time_col not in df.columns:
            raise ValidationError(
                f"time_col '{time_col}' not found in DataFrame",
                context={"available_columns": list(df.columns)},
            )

        sorted_df = df.sort_values(time_col)
        n = len(sorted_df)
        original_positions = sorted_df.index.to_numpy()

        # Minimum usable rows: each fold needs at least 1 test row.
        min_rows = self._n_folds * 2 + self._gap * self._n_folds
        if n < min_rows:
            raise ValidationError(
                "DataFrame too small to form the requested folds",
                context={
                    "n_rows": n,
                    "n_folds": self._n_folds,
                    "min_required": min_rows,
                },
            )

        test_size = n // (self._n_folds + 1)
        folds: list[TimeSeriesFold] = []

        for k in range(self._n_folds):
            train_end = n - (self._n_folds - k) * test_size
            test_start = train_end + self._gap
            test_end = test_start + test_size

            if test_end > n:
                test_end = n

            train_pos = original_positions[:train_end]
            test_pos = original_positions[test_start:test_end]

            folds.append(
                TimeSeriesFold(
                    fold_id=k,
                    train_indices=train_pos,
                    test_indices=test_pos,
                )
            )
            log.debug(
                "cv_fold_created",
                fold_id=k,
                n_train=len(train_pos),
                n_test=len(test_pos),
            )

        return folds

    def cross_validate(
        self,
        model: AnyModel,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str,
    ) -> list[dict[str, float]]:
        """Run walk-forward cross-validation and return per-fold metrics.

        The method detects the model type by checking for the
        ``prepare_sequences`` attribute (LSTM) vs absence
        (RandomForest).  For LSTM models the sequences are prepared
        per fold; for RF models the flat feature matrix is used
        directly.

        Parameters
        ----------
        model:
            A fitted or unfitted ``LifespanPredictor`` or
            ``ConditionClassifier``.
        df:
            Full labelled feature DataFrame.
        feature_cols:
            Ordered list of feature column names.
        label_col:
            Column name for the regression or classification target.

        Returns
        -------
        list[dict[str, float]]
            Per-fold metric dictionaries.  RUL folds include
            ``mae``, ``rmse``, ``r2``.  Health folds include
            ``accuracy``, ``macro_f1``.

        Raises
        ------
        ValidationError
            If label_col or any feature_col is absent.
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )

        missing = set(feature_cols + [label_col]) - set(df.columns)
        if missing:
            raise ValidationError(
                "DataFrame missing columns required for cross-validation",
                context={"missing_columns": sorted(missing)},
            )

        folds = self.split(df)
        fold_metrics: list[dict[str, float]] = []
        is_lstm = hasattr(model, "prepare_sequences")

        for fold in folds:
            train_df = df.loc[fold.train_indices]
            test_df = df.loc[fold.test_indices]

            if is_lstm:
                lstm_model = cast("LifespanPredictor", model)
                if lstm_model._model is None:  # noqa: SLF001
                    lstm_model.build()

                X_train, y_train = lstm_model.prepare_sequences(  # noqa: N806
                    train_df, feature_cols, label_col
                )
                X_test, y_test = lstm_model.prepare_sequences(  # noqa: N806
                    test_df, feature_cols, label_col
                )
                lstm_model.fit(X_train, y_train, X_test, y_test)
                y_pred, _, _ = lstm_model.predict(X_test, mc_passes=1)
                mae = float(mean_absolute_error(y_test, y_pred))
                rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                r2 = float(r2_score(y_test, y_pred))
                metrics: dict[str, float] = {
                    "fold_id": float(fold.fold_id),
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            else:
                rf_model = cast("ConditionClassifier", model)
                X_train = train_df[feature_cols].values.astype(np.float32)  # noqa: N806
                y_train = train_df[label_col].values
                X_test = test_df[feature_cols].values.astype(np.float32)  # noqa: N806
                y_test = test_df[label_col].values

                rf_model.fit(X_train, y_train)
                y_pred_labels, _ = rf_model.predict(X_test)
                acc = float(accuracy_score(y_test, y_pred_labels))
                f1 = float(
                    f1_score(
                        y_test,
                        y_pred_labels,
                        average="macro",
                        zero_division=0,
                    )
                )
                metrics = {
                    "fold_id": float(fold.fold_id),
                    "accuracy": acc,
                    "macro_f1": f1,
                }

            fold_metrics.append(metrics)
            log.info(
                "cv_fold_complete",
                fold_id=fold.fold_id,
                metrics={k: round(v, 4) for k, v in metrics.items()},
            )

        return fold_metrics
