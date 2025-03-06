"""Model version management with zero-downtime hot-swap.

``ModelVersionStore`` maintains two model slots (RUL and health), polls
MLflow for production version updates, and atomically replaces the
active model when a new version is available.

Hot-swap protocol
-----------------
1. Background polling task wakes every ``reload_interval_s`` seconds.
2. Queries MLflow for the latest ``Production`` version of each model.
3. If the version string changed: load the new artifact into a shadow
   variable (not yet visible to request handlers).
4. Run synthetic validation inference on the shadow model.
5. Acquire the asyncio lock and swap shadow → active.
6. Release the lock.

Request handlers call ``get_rul_model()`` / ``get_health_model()`` which
hold the lock only for the duration of a read (lock-free reads via
assignment atomicity).

Usage::

    store = ModelVersionStore(
        mlflow_tracking_uri="http://mlflow:5000",
        rul_model_name="faultscope-lifespan-predictor",
        health_model_name="faultscope-condition-classifier",
        reload_interval_s=60,
    )
    await store.start()
    rul = store.get_rul_model()
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TypedDict

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from faultscope.common.exceptions import ModelLoadError
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

# Feature dimension used for synthetic validation.
_VAL_FEATURE_DIM: int = 24
_VAL_N_SAMPLES: int = 8


class LoadedModel(TypedDict):
    """Metadata bundle for a loaded MLflow model.

    Keys
    ----
    model:
        The loaded mlflow pyfunc model object.
    version:
        Version string from the MLflow Registry.
    loaded_at:
        UTC timestamp when the model was loaded into memory.
    """

    model: object
    version: str
    loaded_at: datetime


class ModelVersionStore:
    """Manages model loading from MLflow Registry with zero-downtime hot-swap.

    Thread-safe via a single ``asyncio.Lock`` that is held only during
    the atomic slot-swap step.  Inference code reads model objects
    without acquiring the lock (Python's GIL makes object reference
    assignment atomic on CPython).

    Parameters
    ----------
    mlflow_tracking_uri:
        URI of the MLflow tracking server.
    rul_model_name:
        Registered model name for the RUL LSTM.
    health_model_name:
        Registered model name for the health RF classifier.
    reload_interval_s:
        Polling interval in seconds.  Default 60.
    """

    def __init__(
        self,
        mlflow_tracking_uri: str,
        rul_model_name: str,
        health_model_name: str,
        reload_interval_s: int = 60,
    ) -> None:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self._client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        self._rul_model_name = rul_model_name
        self._health_model_name = health_model_name
        self._reload_interval_s = reload_interval_s

        self._rul_model: LoadedModel | None = None
        self._health_model: LoadedModel | None = None
        self._swap_lock = asyncio.Lock()
        self._polling_task: asyncio.Task[None] | None = None
        self._started_at: datetime | None = None

    async def start(self) -> None:
        """Load models initially and start the background polling task.

        Raises
        ------
        ModelLoadError
            If the initial load of either model fails.
        """
        _log.info(
            "version_store_starting",
            rul_model=self._rul_model_name,
            health_model=self._health_model_name,
            reload_interval_s=self._reload_interval_s,
        )
        await self._reload_all()
        self._started_at = datetime.now(tz=UTC)
        self._polling_task = asyncio.create_task(
            self._polling_loop(),
            name="model-version-poll",
        )
        _log.info("version_store_started")

    async def stop(self) -> None:
        """Cancel the background polling task and release resources."""
        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
        _log.info("version_store_stopped")

    def get_rul_model(self) -> LoadedModel:
        """Return the currently active RUL model bundle.

        Returns
        -------
        LoadedModel
            TypedDict with ``model``, ``version``, ``loaded_at``.

        Raises
        ------
        ModelLoadError
            If no RUL model has been loaded yet.
        """
        if self._rul_model is None:
            raise ModelLoadError(
                "RUL model is not loaded.  "
                "Ensure ModelVersionStore.start() completed.",
                context={"model_name": self._rul_model_name},
            )
        return self._rul_model

    def get_health_model(self) -> LoadedModel:
        """Return the currently active health model bundle.

        Returns
        -------
        LoadedModel
            TypedDict with ``model``, ``version``, ``loaded_at``.

        Raises
        ------
        ModelLoadError
            If no health model has been loaded yet.
        """
        if self._health_model is None:
            raise ModelLoadError(
                "Health model is not loaded.  "
                "Ensure ModelVersionStore.start() completed.",
                context={"model_name": self._health_model_name},
            )
        return self._health_model

    async def force_reload(self) -> None:
        """Trigger an immediate model reload outside the polling cycle.

        Called by the ``/models/refresh`` endpoint.
        """
        _log.info("version_store_force_reload")
        await self._reload_all()

    def get_status(self) -> dict[str, object]:
        """Return metadata about both loaded models.

        Returns
        -------
        dict[str, object]
            Keys: ``rul_model``, ``health_model``, ``last_reload``.
        """
        rul_info: dict[str, object] = {}
        health_info: dict[str, object] = {}

        if self._rul_model is not None:
            rul_info = {
                "version": self._rul_model["version"],
                "loaded_at": self._rul_model["loaded_at"].isoformat(),
                "model_name": self._rul_model_name,
            }
        if self._health_model is not None:
            health_info = {
                "version": self._health_model["version"],
                "loaded_at": self._health_model["loaded_at"].isoformat(),
                "model_name": self._health_model_name,
            }

        last_reload: str | None = (
            self._started_at.isoformat() if self._started_at else None
        )
        return {
            "rul_model": rul_info,
            "health_model": health_info,
            "last_reload": last_reload,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _polling_loop(self) -> None:
        """Infinite loop that polls MLflow and hot-swaps models."""
        while True:
            await asyncio.sleep(self._reload_interval_s)
            try:
                await self._reload_if_changed()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                _log.error(
                    "version_store_poll_error",
                    error=str(exc),
                    exc_info=True,
                )

    async def _reload_all(self) -> None:
        """Load both models unconditionally."""
        rul = await self._load_production_model(self._rul_model_name)
        health = await self._load_production_model(self._health_model_name)
        async with self._swap_lock:
            self._rul_model = rul
            self._health_model = health
        _log.info(
            "version_store_reload_all_complete",
            rul_version=(rul["version"] if rul else "none"),
            health_version=(health["version"] if health else "none"),
        )

    async def _reload_if_changed(self) -> None:
        """Check for new production versions and hot-swap if found."""
        rul_version = self._get_latest_production_version(self._rul_model_name)
        health_version = self._get_latest_production_version(
            self._health_model_name
        )

        current_rul_ver = (
            self._rul_model["version"] if self._rul_model else None
        )
        current_health_ver = (
            self._health_model["version"] if self._health_model else None
        )

        if rul_version and rul_version != current_rul_ver:
            _log.info(
                "version_store_rul_update_detected",
                old_version=current_rul_ver,
                new_version=rul_version,
            )
            new_rul = await self._load_production_model(self._rul_model_name)
            if new_rul is not None:
                async with self._swap_lock:
                    self._rul_model = new_rul
                _log.info(
                    "version_store_rul_swapped",
                    version=rul_version,
                )

        if health_version and health_version != current_health_ver:
            _log.info(
                "version_store_health_update_detected",
                old_version=current_health_ver,
                new_version=health_version,
            )
            new_health = await self._load_production_model(
                self._health_model_name
            )
            if new_health is not None:
                async with self._swap_lock:
                    self._health_model = new_health
                _log.info(
                    "version_store_health_swapped",
                    version=health_version,
                )

    def _get_latest_production_version(self, model_name: str) -> str | None:
        """Query MLflow for the current production version string."""
        try:
            versions = self._client.search_model_versions(
                filter_string=(
                    f"name='{model_name}' and current_stage='Production'"
                )
            )
            if not versions:
                return None
            return str(max(versions, key=lambda v: int(v.version)).version)
        except Exception as exc:
            _log.warning(
                "version_store_mlflow_query_failed",
                model_name=model_name,
                error=str(exc),
            )
            return None

    async def _load_production_model(
        self, model_name: str
    ) -> LoadedModel | None:
        """Load a Production model from the MLflow Registry.

        Validates the artifact with synthetic inference before returning.

        Parameters
        ----------
        model_name:
            Registered model name.

        Returns
        -------
        LoadedModel | None
            Populated bundle on success, ``None`` if no production
            version exists.

        Raises
        ------
        ModelLoadError
            If the artifact cannot be loaded or validation fails.
        """
        version = self._get_latest_production_version(model_name)
        if version is None:
            _log.warning(
                "version_store_no_production_version",
                model_name=model_name,
            )
            return None

        uri = f"models:/{model_name}/Production"
        _log.info(
            "version_store_loading_model",
            model_name=model_name,
            version=version,
            uri=uri,
        )

        try:
            model = await asyncio.to_thread(mlflow.pyfunc.load_model, uri)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load '{model_name}' from '{uri}': {exc}",
                context={
                    "model_name": model_name,
                    "version": version,
                    "error": str(exc),
                },
            ) from exc

        # Synthetic validation.
        await self._validate(model, model_name, version)

        loaded_at = datetime.now(tz=UTC)
        _log.info(
            "version_store_model_loaded",
            model_name=model_name,
            version=version,
        )
        return LoadedModel(
            model=model,
            version=version,
            loaded_at=loaded_at,
        )

    async def _validate(
        self, model: object, model_name: str, version: str
    ) -> None:
        """Run synthetic inference to confirm the artifact is healthy."""
        rng = np.random.default_rng(seed=7)
        data = pd.DataFrame(
            rng.standard_normal((_VAL_N_SAMPLES, _VAL_FEATURE_DIM)),
            columns=[f"f{i}" for i in range(_VAL_FEATURE_DIM)],
        )
        try:
            await asyncio.to_thread(model.predict, data)  # type: ignore[union-attr, attr-defined]
        except Exception as exc:
            raise ModelLoadError(
                f"Validation inference failed for '{model_name}' "
                f"v{version}: {exc}",
                context={
                    "model_name": model_name,
                    "version": version,
                    "error": str(exc),
                },
            ) from exc

        _log.debug(
            "version_store_validation_passed",
            model_name=model_name,
            version=version,
        )
