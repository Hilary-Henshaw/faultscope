"""Health-check and readiness probe endpoints.

Routes
------
GET /health
    Returns service status, model load state, uptime, and dependency
    health.  Always returns 200 (degraded state is reported in body).

GET /ready
    Returns 200 only when all models are loaded.  Returns 503 when
    models are unavailable.  Used by Kubernetes readiness probes.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter, Request

from faultscope.common.db.engine import check_connection
from faultscope.common.exceptions import ModelLoadError
from faultscope.common.logging import get_logger
from faultscope.inference.api.schemas import HealthResponse
from faultscope.inference.engine.version_store import ModelVersionStore

_log = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def check_health(request: Request) -> HealthResponse:
    """Return comprehensive service health status.

    Checks:
    - Whether RUL and health models are loaded.
    - Database connectivity.
    - Service uptime.

    Returns ``status: "ok"`` when all components are healthy, or
    ``status: "degraded"`` when one or more dependency is unhealthy.
    HTTP status is always 200 so that load balancers receive a
    response; clients must inspect the body.
    """
    version_store: ModelVersionStore = request.app.state.version_store
    startup_time: float = request.app.state.startup_time

    # Check models.
    rul_loaded = False
    health_loaded = False
    try:
        version_store.get_rul_model()
        rul_loaded = True
    except ModelLoadError:
        pass
    try:
        version_store.get_health_model()
        health_loaded = True
    except ModelLoadError:
        pass

    # Check database.
    db_status = "ok"
    try:
        db_ok = await check_connection()
        db_status = "ok" if db_ok else "unreachable"
    except Exception as exc:
        db_status = f"error: {exc}"

    uptime_s = time.monotonic() - startup_time

    overall_status = (
        "ok"
        if (rul_loaded and health_loaded and db_status == "ok")
        else "degraded"
    )

    _log.debug(
        "health_check",
        status=overall_status,
        rul_loaded=rul_loaded,
        health_loaded=health_loaded,
        db_status=db_status,
        uptime_s=round(uptime_s, 1),
    )

    return HealthResponse(
        status=overall_status,
        models_loaded={
            "rul_model": rul_loaded,
            "health_model": health_loaded,
        },
        uptime_s=round(uptime_s, 2),
        dependencies={"database": db_status},
    )


@router.get("/ready")
async def readiness_probe(request: Request) -> dict[str, str]:
    """Kubernetes readiness probe.

    Returns HTTP 200 with ``{"status": "ready"}`` when both models
    are loaded.  Returns HTTP 503 with an error body when models are
    unavailable.

    The inference service should not receive traffic until this
    endpoint returns 200.
    """
    from fastapi import HTTPException

    version_store: ModelVersionStore = request.app.state.version_store

    rul_ok = False
    health_ok = False
    try:
        version_store.get_rul_model()
        rul_ok = True
    except ModelLoadError:
        pass
    try:
        version_store.get_health_model()
        health_ok = True
    except ModelLoadError:
        pass

    if not (rul_ok and health_ok):
        missing = []
        if not rul_ok:
            missing.append("rul_model")
        if not health_ok:
            missing.append("health_model")
        _log.warning(
            "readiness_probe_failed",
            missing_models=missing,
        )
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "missing_models": missing,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )

    return {"status": "ready"}
