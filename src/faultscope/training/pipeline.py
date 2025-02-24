"""Training orchestrator for FaultScope models.

``TrainingOrchestrator`` loads feature snapshots from the versioned
feature store, trains both the LSTM RUL predictor and the Random Forest
health classifier, evaluates them, and registers them in MLflow.
"""

from __future__ import annotations

import asyncio

import numpy as np
import structlog

from faultscope.common.exceptions import ValidationError
from faultscope.features.config import FeaturesConfig
from faultscope.features.store import VersionedFeatureStore
from faultscope.training.config import TrainingConfig
from faultscope.training.evaluator import ModelEvaluator
from faultscope.training.models.condition_classifier import (
    ConditionClassifier,
)
from faultscope.training.models.lifespan_predictor import LifespanPredictor
from faultscope.training.tracker import MLflowTracker

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class TrainingOrchestrator:
    """Orchestrates end-to-end model training for FaultScope.

    Steps:

    1. Load ``train``, ``validation``, and ``test`` feature snapshots
       from ``VersionedFeatureStore``.
    2. Train ``LifespanPredictor`` (LSTM) on the training split,
       using the validation split for early stopping.
    3. Evaluate the RUL model on the test split.
    4. Train ``ConditionClassifier`` (Random Forest) on the training
       split.
    5. Evaluate the health model on the test split.
    6. Log parameters, metrics, and model artefacts to MLflow.
    7. Register both models in the MLflow Model Registry.

    Parameters
    ----------
    config:
        Fully populated ``TrainingConfig`` instance.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._cfg = config
        self._evaluator = ModelEvaluator()
        self._tracker = MLflowTracker(
            tracking_uri=config.mlflow_tracking_uri,
            experiment_name=config.mlflow_experiment_name,
        )

    async def run(
        self,
        dataset_version: str = "v1",
    ) -> dict[str, str]:
        """Run the full training pipeline.

        Parameters
        ----------
        dataset_version:
            Dataset version tag to load from the feature store.

        Returns
        -------
        dict[str, str]
            Mapping with ``"rul_run_id"`` and ``"health_run_id"``
            containing the MLflow run IDs for each trained model.

        Raises
        ------
        ValidationError
            If the feature snapshots are empty or malformed.
        DatabaseError
            If the feature store cannot be reached.
        """
        log.info(
            "training_pipeline_started",
            dataset_version=dataset_version,
            mlflow_experiment=self._cfg.mlflow_experiment_name,
        )

        # Load feature snapshots.
        features_cfg = FeaturesConfig()
        store = VersionedFeatureStore(
            db_url=features_cfg.db_url,
            dataset_version=dataset_version,
        )

        try:
            train_df, val_df, test_df = await asyncio.gather(
                store.load_snapshot("train", dataset_version),
                store.load_snapshot("validation", dataset_version),
                store.load_snapshot("test", dataset_version),
            )
        finally:
            await store.close()

        feature_cols = store.get_feature_columns(train_df)
        if not feature_cols:
            raise ValidationError(
                "No feature columns found in the training snapshot",
                context={"dataset_version": dataset_version},
            )

        log.info(
            "feature_snapshots_loaded",
            n_train=len(train_df),
            n_val=len(val_df),
            n_test=len(test_df),
            n_features=len(feature_cols),
        )

        training_data_info: dict[str, object] = {
            "dataset_version": dataset_version,
            "n_train_rows": len(train_df),
            "n_val_rows": len(val_df),
            "n_test_rows": len(test_df),
            "n_features": len(feature_cols),
        }

        # ── LSTM RUL model ────────────────────────────────────────
        rul_run_id = await asyncio.get_event_loop().run_in_executor(
            None,
            self._train_rul_model,
            train_df,
            val_df,
            test_df,
            feature_cols,
            training_data_info,
        )

        # ── Random Forest health classifier ───────────────────
        health_run_id = await asyncio.get_event_loop().run_in_executor(
            None,
            self._train_health_model,
            train_df,
            test_df,
            feature_cols,
            training_data_info,
        )

        log.info(
            "training_pipeline_complete",
            rul_run_id=rul_run_id,
            health_run_id=health_run_id,
        )
        return {"rul_run_id": rul_run_id, "health_run_id": health_run_id}

    def _train_rul_model(
        self,
        train_df: object,
        val_df: object,
        test_df: object,
        feature_cols: list[str],
        training_data_info: dict[str, object],
    ) -> str:
        """Train, evaluate, and register the LifespanPredictor.

        Returns the MLflow run ID.
        """
        import pandas as pd

        train_df_ = train_df  # type: ignore[assignment]
        val_df_ = val_df  # type: ignore[assignment]
        test_df_ = test_df  # type: ignore[assignment]

        assert isinstance(train_df_, pd.DataFrame)  # noqa: S101
        assert isinstance(val_df_, pd.DataFrame)  # noqa: S101
        assert isinstance(test_df_, pd.DataFrame)  # noqa: S101

        n_features = len(feature_cols)
        predictor = LifespanPredictor(
            sequence_length=self._cfg.sequence_length,
            n_features=n_features,
            lstm_layers=self._cfg.lstm_layers,
            lstm_dropout=self._cfg.lstm_dropout,
            attention_units=self._cfg.attention_units,
            dense_layers=self._cfg.dense_layers,
            dense_dropout=self._cfg.dense_dropout,
            learning_rate=self._cfg.learning_rate,
        )
        predictor.build()

        X_train, y_train = predictor.prepare_sequences(  # noqa: N806
            train_df_, feature_cols, "rul_cycles"
        )
        X_val, y_val = predictor.prepare_sequences(  # noqa: N806
            val_df_, feature_cols, "rul_cycles"
        )
        X_test, y_test = predictor.prepare_sequences(  # noqa: N806
            test_df_, feature_cols, "rul_cycles"
        )

        config_snapshot: dict[str, object] = {
            "sequence_length": self._cfg.sequence_length,
            "lstm_layers": str(self._cfg.lstm_layers),
            "lstm_dropout": self._cfg.lstm_dropout,
            "attention_units": self._cfg.attention_units,
            "dense_layers": str(self._cfg.dense_layers),
            "dense_dropout": self._cfg.dense_dropout,
            "learning_rate": self._cfg.learning_rate,
            "batch_size": self._cfg.batch_size,
            "max_epochs": self._cfg.max_epochs,
            "n_features": n_features,
            "dataset_version": training_data_info.get("dataset_version"),
        }

        run = self._tracker.start_run("lifespan_predictor", config_snapshot)
        run_id: str = run.info.run_id

        try:
            history = predictor.fit(X_train, y_train, X_val, y_val)

            # Log epoch-level validation metrics.
            for epoch, (loss, mae) in enumerate(
                zip(
                    history.history["val_loss"],
                    history.history["val_mae"],
                    strict=False,
                )
            ):
                self._tracker.log_metrics(
                    {"val_loss": float(loss), "val_mae": float(mae)},
                    step=epoch,
                )

            # Evaluate on test split.
            y_pred_mean, y_pred_lower, y_pred_upper = predictor.predict(
                X_test, mc_passes=10
            )
            rul_metrics = self._evaluator.evaluate_rul(y_test, y_pred_mean)
            self._tracker.log_metrics(
                {
                    "test_mae": rul_metrics["mae"],
                    "test_rmse": rul_metrics["rmse"],
                    "test_r2": rul_metrics["r2_score"],
                    "test_mape": rul_metrics["mape"],
                    "test_nasa_score": rul_metrics["nasa_scoring"],
                    "ci_mean_width": float(
                        np.mean(y_pred_upper - y_pred_lower)
                    ),
                }
            )

            # Save and log artefact.
            self._tracker.log_model(predictor, "lifespan_predictor", "model")

            # Generate model card.
            card = self._evaluator.generate_model_card(
                "lifespan_predictor",
                rul_metrics,
                feature_cols,
                training_data_info,
            )
            self._tracker.log_artifact_dict(card, "model_card.pkl")

            # Register in MLflow Registry.
            version = self._tracker.register_model(
                run_id, "faultscope-lifespan-predictor"
            )
            self._tracker.promote_to_production(
                "faultscope-lifespan-predictor", version
            )

        finally:
            self._tracker.end_run()

        log.info(
            "rul_model_training_complete",
            run_id=run_id,
            test_mae=round(rul_metrics["mae"], 4),
            test_rmse=round(rul_metrics["rmse"], 4),
        )
        return run_id

    def _train_health_model(
        self,
        train_df: object,
        test_df: object,
        feature_cols: list[str],
        training_data_info: dict[str, object],
    ) -> str:
        """Train, evaluate, and register the ConditionClassifier.

        Returns the MLflow run ID.
        """
        import pandas as pd

        train_df_ = train_df  # type: ignore[assignment]
        test_df_ = test_df  # type: ignore[assignment]

        assert isinstance(train_df_, pd.DataFrame)  # noqa: S101
        assert isinstance(test_df_, pd.DataFrame)  # noqa: S101

        classifier = ConditionClassifier(
            n_estimators=self._cfg.rf_n_estimators,
            max_depth=self._cfg.rf_max_depth,
            class_weight=self._cfg.rf_class_weight,
        )

        X_train = train_df_[feature_cols].values.astype(np.float32)  # noqa: N806
        y_train = train_df_["health_label"].values
        X_test = test_df_[feature_cols].values.astype(np.float32)  # noqa: N806
        y_test = test_df_["health_label"].values

        config_snapshot: dict[str, object] = {
            "n_estimators": self._cfg.rf_n_estimators,
            "max_depth": self._cfg.rf_max_depth,
            "class_weight": self._cfg.rf_class_weight,
            "n_features": len(feature_cols),
            "dataset_version": training_data_info.get("dataset_version"),
        }

        run = self._tracker.start_run("condition_classifier", config_snapshot)
        run_id: str = run.info.run_id

        try:
            classifier.fit(X_train, y_train)

            y_pred_labels, y_proba = classifier.predict(X_test)
            health_metrics = self._evaluator.evaluate_health(
                y_test, y_pred_labels, y_proba
            )
            self._tracker.log_metrics(
                {
                    "test_accuracy": health_metrics["accuracy"],
                    "test_macro_f1": health_metrics["macro_f1"],
                    "test_weighted_f1": health_metrics["weighted_f1"],
                    "test_imminent_failure_recall": health_metrics[
                        "imminent_failure_recall"
                    ],
                }
            )

            # Feature importances as metrics.
            importances = classifier.feature_importances(
                feature_cols, top_k=20
            )
            self._tracker.log_metrics(
                {
                    f"feat_imp__{k}": v
                    for k, v in list(importances.items())[:20]
                }
            )

            self._tracker.log_model(
                classifier, "condition_classifier", "model"
            )

            card = self._evaluator.generate_model_card(
                "condition_classifier",
                health_metrics,
                feature_cols,
                training_data_info,
            )
            self._tracker.log_artifact_dict(card, "model_card.pkl")

            version = self._tracker.register_model(
                run_id, "faultscope-condition-classifier"
            )
            self._tracker.promote_to_production(
                "faultscope-condition-classifier", version
            )

        finally:
            self._tracker.end_run()

        log.info(
            "health_model_training_complete",
            run_id=run_id,
            test_accuracy=round(health_metrics["accuracy"], 4),
            test_macro_f1=round(health_metrics["macro_f1"], 4),
            imminent_failure_recall=round(
                health_metrics["imminent_failure_recall"], 4
            ),
        )
        return run_id
