"""Model catalog endpoints.

Routes
------
GET /api/v1/models
    Returns metadata about currently loaded models (version, loaded_at).

POST /api/v1/models/refresh
    Forces an immediate model reload from the MLflow Registry without
    waiting for the next polling cycle.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request

from faultscope.common.logging import get_logger
from faultscope.inference.api.schemas import ModelCatalogResponse
from faultscope.inference.engine.version_store import ModelVersionStore

_log = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Models"])


@router.get("/models", response_model=ModelCatalogResponse)
async def list_models(request: Request) -> ModelCatalogResponse:
    """Return metadata about the currently active models.

    Includes model version, MLflow model name, and the timestamp at
    which each model was loaded into memory.

    Returns
    -------
    ModelCatalogResponse
        Populated with ``rul_model``, ``health_model``, and
        ``last_reload``.
    """
    version_store: ModelVersionStore = request.app.state.version_store
    status = version_store.get_status()

    rul_info: dict[str, object] = status.get(  # type: ignore[assignment]
        "rul_model", {}
    )
    health_info: dict[str, object] = status.get(  # type: ignore[assignment]
        "health_model", {}
    )
    last_reload_raw = status.get("last_reload")

    if last_reload_raw is not None and isinstance(last_reload_raw, str):
        last_reload = datetime.fromisoformat(last_reload_raw)
    else:
        last_reload = datetime.now(tz=UTC)

    _log.debug(
        "model_catalog_requested",
        rul_version=rul_info.get("version"),
        health_version=health_info.get("version"),
    )

    return ModelCatalogResponse(
        rul_model=rul_info,
        health_model=health_info,
        last_reload=last_reload,
    )


@router.post("/models/refresh")
async def refresh_models(request: Request) -> dict[str, str]:
    """Force an immediate model reload from the MLflow Registry.

    This endpoint bypasses the polling interval and triggers
    synchronous re-loading of both models.  Use it after promoting
    a new model version and wanting zero-delay activation.

    Returns
    -------
    dict[str, str]
        ``{"status": "reloaded", "timestamp": "<ISO-8601>"}``
    """
    version_store: ModelVersionStore = request.app.state.version_store

    _log.info("model_refresh_triggered_via_api")
    await version_store.force_reload()

    ts = datetime.now(tz=UTC).isoformat()
    _log.info("model_refresh_complete", timestamp=ts)

    return {"status": "reloaded", "timestamp": ts}
