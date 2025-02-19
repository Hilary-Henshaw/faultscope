"""MLflow experiment tracking wrapper for FaultScope training.

``MLflowTracker`` abstracts all MLflow calls behind a clean interface
that logs parameters, metrics, model artefacts, and handles Registry
stage transitions.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import structlog
from mlflow.tracking import MlflowClient

from faultscope.common.exceptions import ValidationError

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_VALID_MODEL_TYPES: frozenset[str] = frozenset(
    {"lifespan_predictor", "condition_classifier"}
)


class MLflowTracker:
    """Wraps MLflow for faultscope training experiments.

    Handles experiment creation/retrieval, parameter and metric
    logging, model artefact registration, and stage promotion.

    Parameters
    ----------
    tracking_uri:
        URI of the MLflow tracking server, e.g.
        ``"http://localhost:5000"``.
    experiment_name:
        Name of the MLflow experiment; created if it does not exist.
    """

    def __init__(
        self,
        tracking_uri: str,
        experiment_name: str,
    ) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._client: MlflowClient = MlflowClient(tracking_uri)
        self._experiment_name = experiment_name
        self._experiment_id: str = self._get_or_create_experiment()
        self._active_run: mlflow.ActiveRun | None = None

    def _get_or_create_experiment(self) -> str:
        """Return experiment ID, creating the experiment if needed."""
        experiment = mlflow.get_experiment_by_name(self._experiment_name)
        if experiment is not None:
            return str(experiment.experiment_id)
        exp_id = mlflow.create_experiment(self._experiment_name)
        log.info(
            "mlflow_experiment_created",
            name=self._experiment_name,
            experiment_id=exp_id,
        )
        return exp_id

    def start_run(
        self,
        model_type: str,
        config_snapshot: dict[str, object],
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run and log all config parameters.

        Parameters
        ----------
        model_type:
            One of ``"lifespan_predictor"`` or
            ``"condition_classifier"``.
        config_snapshot:
            Dictionary of training hyperparameters to log.

        Returns
        -------
        mlflow.ActiveRun
            The active MLflow run context manager.

        Raises
        ------
        ValidationError
            If *model_type* is not a recognised model type.
        """
        if model_type not in _VALID_MODEL_TYPES:
            raise ValidationError(
                f"Unknown model_type '{model_type}'",
                context={
                    "model_type": model_type,
                    "valid_types": sorted(_VALID_MODEL_TYPES),
                },
            )

        run: mlflow.ActiveRun = mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=f"{model_type}",
            tags={"model_type": model_type},
        )
        self._active_run = run

        # Log each config value as a flat parameter string.
        flat_params: dict[str, str] = {
            str(k): str(v) for k, v in config_snapshot.items()
        }
        mlflow.log_params(flat_params)

        log.info(
            "mlflow_run_started",
            run_id=run.info.run_id,
            model_type=model_type,
            experiment=self._experiment_name,
        )
        return run

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log a dict of scalar metrics to the active MLflow run.

        Parameters
        ----------
        metrics:
            Mapping of metric name to float value.
        step:
            Optional training step / epoch counter.
        """
        mlflow.log_metrics(metrics, step=step)
        log.debug(
            "mlflow_metrics_logged",
            metrics={k: round(v, 4) for k, v in metrics.items()},
            step=step,
        )

    def log_model(
        self,
        model: object,
        model_type: str,
        artifact_path: str,
    ) -> str:
        """Save model artifact to MLflow and return the run ID.

        For ``lifespan_predictor`` the Keras SavedModel format is used;
        for ``condition_classifier`` pickle-based sklearn logging is
        used.

        Parameters
        ----------
        model:
            The trained model object.
        model_type:
            ``"lifespan_predictor"`` or ``"condition_classifier"``.
        artifact_path:
            Sub-path within the run's artifact store.

        Returns
        -------
        str
            Active run ID.

        Raises
        ------
        ValidationError
            If no run is active or model_type is unrecognised.
        """
        if self._active_run is None:
            raise ValidationError(
                "No active MLflow run; call start_run first",
                context={},
            )
        if model_type not in _VALID_MODEL_TYPES:
            raise ValidationError(
                f"Unknown model_type '{model_type}'",
                context={"model_type": model_type},
            )

        run_id: str = self._active_run.info.run_id

        if model_type == "lifespan_predictor":
            # Save Keras model to a temp dir then log as artefact.
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "keras_model"
                model.save(str(save_path))  # type: ignore[union-attr, attr-defined]
                mlflow.log_artifacts(
                    str(save_path), artifact_path=artifact_path
                )
        else:
            # Use MLflow sklearn integration for the RF model.
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
            )

        log.info(
            "mlflow_model_logged",
            run_id=run_id,
            model_type=model_type,
            artifact_path=artifact_path,
        )
        return run_id

    def register_model(
        self,
        run_id: str,
        model_name: str,
    ) -> str:
        """Register the model in the MLflow Model Registry.

        Parameters
        ----------
        run_id:
            MLflow run ID whose artefacts contain the model.
        model_name:
            Registered model name in the MLflow Registry.

        Returns
        -------
        str
            The new model version string.
        """
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
        )
        version: str = str(result.version)
        log.info(
            "mlflow_model_registered",
            model_name=model_name,
            version=version,
            run_id=run_id,
        )
        return version

    def promote_to_production(
        self,
        model_name: str,
        version: str,
    ) -> None:
        """Transition a model version to the Production stage.

        Any existing Production versions of the same model are
        transitioned to Archived before the new version is promoted.

        Parameters
        ----------
        model_name:
            Registered model name.
        version:
            Version string to promote.
        """
        # Archive any currently live production versions.
        versions = self._client.get_latest_versions(
            model_name, stages=["Production"]
        )
        for v in versions:
            if v.version != version:
                self._client.transition_model_version_stage(
                    name=model_name,
                    version=v.version,
                    stage="Archived",
                    archive_existing_versions=False,
                )
                log.info(
                    "mlflow_version_archived",
                    model_name=model_name,
                    version=v.version,
                )

        self._client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=False,
        )
        log.info(
            "mlflow_version_promoted",
            model_name=model_name,
            version=version,
            stage="Production",
        )

    def end_run(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()
        self._active_run = None
        log.info("mlflow_run_ended")

    def log_artifact_dict(
        self,
        data: dict[str, Any],
        filename: str,
    ) -> None:
        """Pickle *data* and upload it as an MLflow artefact.

        Parameters
        ----------
        data:
            Arbitrary serialisable dictionary (e.g. model card).
        filename:
            Filename for the artefact inside the run.
        """
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            pickle.dump(data, tmp)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="metadata")
        log.debug("mlflow_artifact_dict_logged", filename=filename)
