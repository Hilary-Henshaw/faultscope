"""Random Forest health-status classifier.

``ConditionClassifier`` wraps scikit-learn's ``RandomForestClassifier``
with a ``LabelEncoder`` for the four health states and provides SHAP-
free feature importance via mean-decrease-in-impurity (MDI).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import structlog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from faultscope.common.exceptions import ModelLoadError, ValidationError

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class ConditionClassifier:
    """Random Forest classifier for equipment health status.

    Predicts one of: ``healthy`` | ``degrading`` | ``critical`` |
    ``imminent_failure``.  Class weights are balanced to address the
    natural label imbalance in run-to-failure datasets.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum depth of each tree.  ``None`` for unlimited depth.
    class_weight:
        Strategy for handling class imbalance.  Accepts
        ``"balanced"`` or ``"balanced_subsample"``.
    """

    HEALTH_LABELS: ClassVar[list[str]] = [
        "healthy",
        "degrading",
        "critical",
        "imminent_failure",
    ]

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        class_weight: str = "balanced",
    ) -> None:
        if n_estimators < 1:
            raise ValidationError(
                "n_estimators must be a positive integer",
                context={"n_estimators": n_estimators},
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight

        self._encoder = LabelEncoder()
        self._encoder.fit(self.HEALTH_LABELS)

        self._clf: RandomForestClassifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,  # type: ignore[arg-type]
            n_jobs=-1,
            random_state=42,
            oob_score=True,
        )
        self._is_fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Train the Random Forest classifier.

        Parameters
        ----------
        X_train:
            Feature matrix of shape ``(n_samples, n_features)``.
        y_train:
            String health labels of shape ``(n_samples,)``.

        Raises
        ------
        ValidationError
            If *y_train* contains labels not in ``HEALTH_LABELS``.
        """
        unknown = set(y_train.tolist()) - set(self.HEALTH_LABELS)
        if unknown:
            raise ValidationError(
                "y_train contains unrecognised health labels",
                context={"unknown_labels": sorted(unknown)},
            )

        y_encoded = self._encoder.transform(y_train)
        self._clf.fit(X_train, y_encoded)
        self._is_fitted = True

        oob = float(self._clf.oob_score_)
        log.info(
            "condition_classifier_fitted",
            n_samples=len(X_train),
            n_features=X_train.shape[1],
            oob_score=round(oob, 4),
            n_estimators=self.n_estimators,
        )

    def predict(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict health labels and class probabilities.

        Parameters
        ----------
        X:
            Feature matrix of shape ``(n_samples, n_features)``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(predicted_labels, probabilities_matrix)`` where
            ``predicted_labels`` is a string array of shape
            ``(n_samples,)`` and ``probabilities_matrix`` has shape
            ``(n_samples, n_classes)`` with columns in the same order
            as ``HEALTH_LABELS``.

        Raises
        ------
        ValidationError
            If the classifier has not been fitted.
        """
        if not self._is_fitted:
            raise ValidationError(
                "Classifier must be fitted before calling predict",
                context={},
            )

        y_encoded: np.ndarray = self._clf.predict(X)
        labels: np.ndarray = self._encoder.inverse_transform(y_encoded)

        # Probabilities are returned in the order of
        # self._clf.classes_ which may differ from HEALTH_LABELS.
        raw_proba: np.ndarray = self._clf.predict_proba(X)

        # Re-order columns to match HEALTH_LABELS order.
        target_order = self._encoder.transform(self.HEALTH_LABELS)
        proba = raw_proba[:, target_order]

        return labels, proba

    def feature_importances(
        self,
        feature_names: list[str],
        top_k: int = 20,
    ) -> dict[str, float]:
        """Return the top-k features by mean decrease in impurity.

        Parameters
        ----------
        feature_names:
            List of feature column names aligned with the training
            matrix columns.
        top_k:
            Maximum number of features to return.

        Returns
        -------
        dict[str, float]
            Mapping of feature name to MDI importance, sorted
            descending by importance, limited to *top_k* entries.

        Raises
        ------
        ValidationError
            If the classifier has not been fitted or the length of
            *feature_names* does not match the model.
        """
        if not self._is_fitted:
            raise ValidationError(
                "Classifier must be fitted before computing importances",
                context={},
            )

        n_model_features = len(self._clf.feature_importances_)
        if len(feature_names) != n_model_features:
            raise ValidationError(
                "feature_names length does not match model feature count",
                context={
                    "expected": n_model_features,
                    "got": len(feature_names),
                },
            )

        importances: np.ndarray = self._clf.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        result: dict[str, float] = {}
        for i in sorted_idx[:top_k]:
            result[feature_names[i]] = float(importances[i])
        return result

    def save(self, path: str) -> None:
        """Persist the classifier and label encoder to disk.

        Uses pickle for both objects; both are written to *path*
        as ``classifier.pkl`` and ``encoder.pkl``.

        Parameters
        ----------
        path:
            Directory path to write artifacts to.

        Raises
        ------
        ValidationError
            If the classifier has not been fitted.
        """
        if not self._is_fitted:
            raise ValidationError(
                "Nothing to save — classifier has not been fitted",
                context={"path": path},
            )

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "classifier.pkl", "wb") as fh:
            pickle.dump(self._clf, fh)
        with open(save_dir / "encoder.pkl", "wb") as fh:
            pickle.dump(self._encoder, fh)

        meta = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "class_weight": self.class_weight,
        }
        with open(save_dir / "meta.pkl", "wb") as fh:
            pickle.dump(meta, fh)

        log.info("condition_classifier_saved", path=path)

    @classmethod
    def load(cls, path: str) -> ConditionClassifier:
        """Load a previously saved ``ConditionClassifier``.

        Parameters
        ----------
        path:
            Directory path written by ``save``.

        Returns
        -------
        ConditionClassifier
            Reconstructed classifier ready for inference.

        Raises
        ------
        ModelLoadError
            If any of the artifact files cannot be loaded.
        """
        save_dir = Path(path)
        for fname in ("classifier.pkl", "encoder.pkl", "meta.pkl"):
            if not (save_dir / fname).exists():
                raise ModelLoadError(
                    f"{fname} not found in artifact directory",
                    context={"path": path},
                )

        try:
            with open(save_dir / "meta.pkl", "rb") as fh:
                meta: dict[str, Any] = pickle.load(fh)  # noqa: S301  # nosec B301
            with open(save_dir / "classifier.pkl", "rb") as fh:
                clf: RandomForestClassifier = pickle.load(fh)  # noqa: S301  # nosec B301
            with open(save_dir / "encoder.pkl", "rb") as fh:
                encoder: LabelEncoder = pickle.load(fh)  # noqa: S301  # nosec B301
        except Exception as exc:
            raise ModelLoadError(
                "Failed to deserialise ConditionClassifier artifacts",
                context={"path": path, "error": str(exc)},
            ) from exc

        instance = cls(
            n_estimators=int(meta["n_estimators"]),
            max_depth=int(meta["max_depth"]),
            class_weight=str(meta["class_weight"]),
        )
        instance._clf = clf  # noqa: SLF001
        instance._encoder = encoder  # noqa: SLF001
        instance._is_fitted = True  # noqa: SLF001

        log.info("condition_classifier_loaded", path=path)
        return instance
