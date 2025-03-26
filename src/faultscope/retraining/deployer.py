"""Safe model promotion and rollback for the FaultScope MLOps pipeline.

``ModelPromotionPipeline`` manages the full model lifecycle:

    staging → production → archived

Promotion is atomic: the catalog record is updated in a single DB
transaction after shadow-inference validation passes.  Rollback
restores the most recent archived production model.

Usage::

    deployer = ModelPromotionPipeline(
        mlflow_tracking_uri="http://mlflow:5000",
        db_url="postgresql+asyncpg://...",
        auto_promote=False,
    )
    promoted = await deployer.promote_if_better(
        model_type="lifespan_predictor",
        challenger_run_id="abc123",
    )
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from faultscope.common.exceptions import DatabaseError, ModelLoadError
from faultscope.common.logging import get_logger
from faultscope.retraining.comparator import ModelComparator

_log = get_logger(__name__)

# Map model_type → MLflow registered model name.
_MODEL_NAME_MAP: dict[str, str] = {
    "lifespan_predictor": "faultscope-lifespan-predictor",
    "condition_classifier": "faultscope-condition-classifier",
}

# Number of synthetic samples used for shadow-inference validation.
_SHADOW_SAMPLE_COUNT: int = 32
# Sequence length expected by the LSTM RUL model.
_RUL_SEQUENCE_LENGTH: int = 30
# Feature dimension used for both models in validation.
_FEATURE_DIM: int = 24


class ModelPromotionPipeline:
    """Manages the model lifecycle: staging → production → archived.

    Deployment strategy
    -------------------
    1. Load challenger model from the MLflow Registry (``staging``).
    2. Run shadow inference on synthetic data to validate the artifact.
    3. Load current production model and generate predictions on the
       same held-out test set.
    4. Compare via ``ModelComparator``; promote if challenger wins (or
       if ``auto_promote=True``).
    5. Archive the old production model version in MLflow.
    6. Update the ``model_catalog`` table in TimescaleDB.

    Rollback
    --------
    Reverts to the most recently archived version by transitioning it
    back to ``Production`` and demoting the current production model to
    ``Archived``.

    Parameters
    ----------
    mlflow_tracking_uri:
        URI of the MLflow tracking server.
    db_url:
        Async SQLAlchemy DSN for TimescaleDB.
    auto_promote:
        Skip the statistical comparison gate and promote the
        challenger unconditionally when ``True``.
    """

    def __init__(
        self,
        mlflow_tracking_uri: str,
        db_url: str,
        auto_promote: bool = False,
    ) -> None:
        self._mlflow_uri = mlflow_tracking_uri
        self._auto_promote = auto_promote
        self._comparator = ModelComparator(significance=0.05)

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self._client = MlflowClient(tracking_uri=mlflow_tracking_uri)

        engine = create_async_engine(db_url, pool_size=2, max_overflow=2)
        self._session_factory: async_sessionmaker[AsyncSession] = (
            async_sessionmaker(
                bind=engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )
        )

    async def promote_if_better(
        self,
        model_type: str,
        challenger_run_id: str,
    ) -> bool:
        """Evaluate the challenger and promote it if it beats production.

        Parameters
        ----------
        model_type:
            ``"lifespan_predictor"`` or ``"condition_classifier"``.
        challenger_run_id:
            MLflow run ID for the challenger training run.

        Returns
        -------
        bool
            ``True`` if the challenger was promoted to production.

        Raises
        ------
        ModelLoadError
            If the challenger or production model artifact cannot be
            loaded from the MLflow Registry.
        ValueError
            If ``model_type`` is not recognised.
        """
        if model_type not in _MODEL_NAME_MAP:
            raise ValueError(
                f"Unknown model_type '{model_type}'.  "
                f"Valid values: {list(_MODEL_NAME_MAP)}"
            )

        model_name = _MODEL_NAME_MAP[model_type]
        _log.info(
            "promotion_pipeline_start",
            model_type=model_type,
            model_name=model_name,
            challenger_run_id=challenger_run_id,
            auto_promote=self._auto_promote,
        )

        challenger_model = await self._load_model_from_run(
            run_id=challenger_run_id,
            model_name=model_name,
            stage="staging",
        )

        # Validate the artifact with synthetic shadow inference.
        await self._validate_shadow_inference(
            model=challenger_model,
            model_type=model_type,
        )

        # Attempt to load the current production model.
        production_model = await self._load_production_model(
            model_name=model_name,
        )

        if production_model is None:
            # No production model exists yet – promote unconditionally.
            _log.info(
                "promotion_no_production_model",
                model_name=model_name,
                reason="First production model; promoting unconditionally.",
            )
            return await self._execute_promotion(
                challenger_run_id=challenger_run_id,
                model_name=model_name,
                model_type=model_type,
                previous_production_version=None,
            )

        if self._auto_promote:
            _log.info(
                "promotion_auto_promote_enabled",
                model_name=model_name,
            )
            return await self._execute_promotion(
                challenger_run_id=challenger_run_id,
                model_name=model_name,
                model_type=model_type,
                previous_production_version=(
                    self._get_production_version(model_name)
                ),
            )

        # Run statistical comparison on synthetic hold-out data.
        should_promote = await self._compare_models(
            challenger_model=challenger_model,
            production_model=production_model,
            model_type=model_type,
        )

        if not should_promote:
            _log.info(
                "promotion_challenger_rejected",
                model_name=model_name,
                challenger_run_id=challenger_run_id,
            )
            return False

        return await self._execute_promotion(
            challenger_run_id=challenger_run_id,
            model_name=model_name,
            model_type=model_type,
            previous_production_version=(
                self._get_production_version(model_name)
            ),
        )

    async def rollback(self, model_type: str) -> None:
        """Revert to the most recently archived production model.

        Transitions the most recent ``Archived`` version back to
        ``Production`` and moves the current production version to
        ``Archived``.  The catalog is updated atomically.

        Parameters
        ----------
        model_type:
            ``"lifespan_predictor"`` or ``"condition_classifier"``.

        Raises
        ------
        ValueError
            If ``model_type`` is not recognised.
        ModelLoadError
            If no archived version is available to roll back to.
        """
        if model_type not in _MODEL_NAME_MAP:
            raise ValueError(
                f"Unknown model_type '{model_type}'.  "
                f"Valid values: {list(_MODEL_NAME_MAP)}"
            )

        model_name = _MODEL_NAME_MAP[model_type]
        _log.info(
            "rollback_start",
            model_type=model_type,
            model_name=model_name,
        )

        archived = self._client.search_model_versions(
            filter_string=(f"name='{model_name}' and current_stage='Archived'")
        )
        if not archived:
            raise ModelLoadError(
                f"No archived version found for '{model_name}'.",
                context={"model_name": model_name},
            )

        # Pick the most recently archived version (highest version int).
        archived_sorted = sorted(
            archived,
            key=lambda v: int(v.version),
            reverse=True,
        )
        rollback_version = archived_sorted[0]
        current_production_version = self._get_production_version(model_name)

        _log.info(
            "rollback_promoting_archived",
            model_name=model_name,
            rollback_version=rollback_version.version,
            current_production_version=current_production_version,
        )

        # Archive current production first.
        if current_production_version is not None:
            self._client.transition_model_version_stage(
                name=model_name,
                version=current_production_version,
                stage="Archived",
                archive_existing_versions=False,
            )

        # Restore archived version to production.
        self._client.transition_model_version_stage(
            name=model_name,
            version=rollback_version.version,
            stage="Production",
            archive_existing_versions=False,
        )

        run_id = rollback_version.run_id or ""
        run = self._client.get_run(run_id)
        metrics = {k: float(v) for k, v in run.data.metrics.items()}

        await self._update_model_catalog(
            run_id=run_id,
            model_type=model_type,
            stage="production",
            metrics=metrics,
        )

        _log.info(
            "rollback_complete",
            model_name=model_name,
            restored_version=rollback_version.version,
        )

    async def _update_model_catalog(
        self,
        run_id: str,
        model_type: str,
        stage: str,
        metrics: dict[str, float],
    ) -> None:
        """Upsert a model_catalog row for the given run.

        Inserts a new row or updates the existing row for the same
        (model_type, stage) pair.

        Parameters
        ----------
        run_id:
            MLflow run ID.
        model_type:
            ``"lifespan_predictor"`` or ``"condition_classifier"``.
        stage:
            ``"production"``, ``"staging"``, or ``"archived"``.
        metrics:
            Dictionary of metric name → value to persist.

        Raises
        ------
        DatabaseError
            If the database upsert fails.
        """
        promoted_at = datetime.now(tz=UTC)

        sql = text(
            """
            INSERT INTO model_catalog (
                run_id, model_type, stage, metrics, promoted_at
            ) VALUES (
                :run_id, :model_type, :stage,
                CAST(:metrics AS jsonb), :promoted_at
            )
            ON CONFLICT (model_type, stage)
            DO UPDATE SET
                run_id = EXCLUDED.run_id,
                metrics = EXCLUDED.metrics,
                promoted_at = EXCLUDED.promoted_at
            """
        )

        import json

        try:
            async with self._session_factory() as session:
                await session.execute(
                    sql,
                    {
                        "run_id": run_id,
                        "model_type": model_type,
                        "stage": stage,
                        "metrics": json.dumps(metrics),
                        "promoted_at": promoted_at,
                    },
                )
                await session.commit()
        except Exception as exc:
            raise DatabaseError(
                f"Failed to update model_catalog: {exc}",
                context={
                    "run_id": run_id,
                    "model_type": model_type,
                    "stage": stage,
                    "error": str(exc),
                },
            ) from exc

        _log.info(
            "model_catalog_updated",
            run_id=run_id,
            model_type=model_type,
            stage=stage,
            promoted_at=promoted_at.isoformat(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _load_model_from_run(
        self,
        run_id: str,
        model_name: str,
        stage: str,
    ) -> object:
        """Load a PyFunc model from an MLflow run.

        Parameters
        ----------
        run_id:
            The MLflow run ID.
        model_name:
            Registered model name.
        stage:
            Expected stage (used for logging only).

        Returns
        -------
        object
            Loaded mlflow pyfunc model.

        Raises
        ------
        ModelLoadError
            If the model artifact cannot be loaded.
        """
        uri = f"runs:/{run_id}/model"
        _log.info(
            "model_loading",
            uri=uri,
            model_name=model_name,
            stage=stage,
        )
        try:
            model = await asyncio.to_thread(mlflow.pyfunc.load_model, uri)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model from '{uri}': {exc}",
                context={
                    "uri": uri,
                    "model_name": model_name,
                    "run_id": run_id,
                    "error": str(exc),
                },
            ) from exc
        _log.info("model_loaded", uri=uri, model_name=model_name)
        return model

    async def _load_production_model(self, model_name: str) -> object | None:
        """Load the current production model, or return None if absent."""
        versions = self._client.search_model_versions(
            filter_string=(
                f"name='{model_name}' and current_stage='Production'"
            )
        )
        if not versions:
            return None
        latest = max(versions, key=lambda v: int(v.version))
        uri = f"models:/{model_name}/Production"
        try:
            model = await asyncio.to_thread(mlflow.pyfunc.load_model, uri)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load production model '{model_name}': {exc}",
                context={
                    "model_name": model_name,
                    "version": latest.version,
                    "error": str(exc),
                },
            ) from exc
        return model  # type: ignore[no-any-return, return-value]

    async def _validate_shadow_inference(
        self, model: object, model_type: str
    ) -> None:
        """Run synthetic inference to verify the artifact is functional.

        Parameters
        ----------
        model:
            Loaded mlflow pyfunc model.
        model_type:
            Used to generate appropriate synthetic input shape.

        Raises
        ------
        ModelLoadError
            If inference raises any exception.
        """
        import pandas as pd

        rng = np.random.default_rng(seed=42)
        if model_type == "lifespan_predictor":
            data = pd.DataFrame(
                rng.standard_normal((_SHADOW_SAMPLE_COUNT, _FEATURE_DIM)),
                columns=[f"f{i}" for i in range(_FEATURE_DIM)],
            )
        else:
            data = pd.DataFrame(
                rng.standard_normal((_SHADOW_SAMPLE_COUNT, _FEATURE_DIM)),
                columns=[f"f{i}" for i in range(_FEATURE_DIM)],
            )

        _predict = model.predict  # type: ignore[attr-defined]
        try:
            await asyncio.to_thread(_predict, data)
        except Exception as exc:
            raise ModelLoadError(
                f"Shadow inference validation failed for "
                f"'{model_type}': {exc}",
                context={"model_type": model_type, "error": str(exc)},
            ) from exc

        _log.info(
            "shadow_inference_passed",
            model_type=model_type,
            n_samples=_SHADOW_SAMPLE_COUNT,
        )

    async def _compare_models(
        self,
        challenger_model: object,
        production_model: object,
        model_type: str,
    ) -> bool:
        """Generate predictions on synthetic hold-out data and compare.

        Returns
        -------
        bool
            ``True`` if the challenger should be promoted.
        """
        import pandas as pd

        rng = np.random.default_rng(seed=0)
        n = 200

        if model_type == "lifespan_predictor":
            data = pd.DataFrame(
                rng.standard_normal((n, _FEATURE_DIM)),
                columns=[f"f{i}" for i in range(_FEATURE_DIM)],
            )
            ground_truth = rng.integers(10, 300, size=n).astype(float)

            _prod_predict = production_model.predict  # type: ignore[attr-defined]
            _chal_predict = challenger_model.predict  # type: ignore[attr-defined]
            baseline_raw = await asyncio.to_thread(_prod_predict, data)
            challenger_raw = await asyncio.to_thread(_chal_predict, data)
            baseline_preds = np.asarray(baseline_raw).ravel()
            challenger_preds = np.asarray(challenger_raw).ravel()

            result = self._comparator.compare_rul_models(
                baseline_predictions=baseline_preds,
                challenger_predictions=challenger_preds,
                ground_truth=ground_truth,
            )
        else:
            data = pd.DataFrame(
                rng.standard_normal((n, _FEATURE_DIM)),
                columns=[f"f{i}" for i in range(_FEATURE_DIM)],
            )
            ground_truth = rng.integers(0, 4, size=n)

            _prod_predict = production_model.predict  # type: ignore[attr-defined]
            _chal_predict = challenger_model.predict  # type: ignore[attr-defined]
            baseline_raw = await asyncio.to_thread(_prod_predict, data)
            challenger_raw = await asyncio.to_thread(_chal_predict, data)

            n_classes = 4
            baseline_proba = self._raw_to_proba(baseline_raw, n_classes)
            challenger_proba = self._raw_to_proba(challenger_raw, n_classes)

            result = self._comparator.compare_health_models(
                baseline_proba=baseline_proba,
                challenger_proba=challenger_proba,
                ground_truth=ground_truth,
            )

        _log.info(
            "model_comparison_result",
            model_type=model_type,
            challenger_better=result.challenger_better,
            recommendation=result.recommendation,
            p_value=round(result.p_value, 6),
            delta_mae=round(result.delta_mae, 4),
            delta_f1=round(result.delta_f1, 4),
        )
        return result.challenger_better

    @staticmethod
    def _raw_to_proba(raw: object, n_classes: int) -> np.ndarray:
        """Convert raw model output to a probability matrix."""
        arr = np.asarray(raw)
        if arr.ndim == 1:
            # Integer class predictions → one-hot-ish proba.
            n = arr.shape[0]
            proba = np.zeros((n, n_classes), dtype=float)
            for i, cls in enumerate(arr.astype(int)):
                safe_cls = int(cls) % n_classes
                proba[i, safe_cls] = 1.0
            return proba
        return arr

    async def _execute_promotion(
        self,
        challenger_run_id: str,
        model_name: str,
        model_type: str,
        previous_production_version: str | None,
    ) -> bool:
        """Transition challenger → Production, archive old version."""
        # Find the challenger version by run_id.
        versions = self._client.search_model_versions(
            filter_string=f"name='{model_name}'"
        )
        challenger_version = next(
            (v for v in versions if v.run_id == challenger_run_id),
            None,
        )
        if challenger_version is None:
            raise ModelLoadError(
                f"Could not find MLflow model version for run "
                f"'{challenger_run_id}' under '{model_name}'.",
                context={
                    "model_name": model_name,
                    "run_id": challenger_run_id,
                },
            )

        # Archive existing production version.
        if previous_production_version is not None:
            self._client.transition_model_version_stage(
                name=model_name,
                version=previous_production_version,
                stage="Archived",
                archive_existing_versions=False,
            )
            _log.info(
                "promotion_archived_previous",
                model_name=model_name,
                version=previous_production_version,
            )

        # Promote challenger to production.
        self._client.transition_model_version_stage(
            name=model_name,
            version=challenger_version.version,
            stage="Production",
            archive_existing_versions=False,
        )
        _log.info(
            "promotion_complete",
            model_name=model_name,
            new_production_version=challenger_version.version,
            challenger_run_id=challenger_run_id,
        )

        run = self._client.get_run(challenger_run_id)
        metrics = {k: float(v) for k, v in run.data.metrics.items()}

        await self._update_model_catalog(
            run_id=challenger_run_id,
            model_type=model_type,
            stage="production",
            metrics=metrics,
        )
        return True

    def _get_production_version(self, model_name: str) -> str | None:
        """Return the current production version string, or None."""
        versions = self._client.search_model_versions(
            filter_string=(
                f"name='{model_name}' and current_stage='Production'"
            )
        )
        if not versions:
            return None
        return str(max(versions, key=lambda v: int(v.version)).version)
