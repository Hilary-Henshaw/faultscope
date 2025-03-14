"""Incident lifecycle REST endpoints for the FaultScope alerting service.

Routes
------
POST   /api/v1/incidents/evaluate          Submit a prediction for evaluation
GET    /api/v1/incidents                   List / filter incidents
POST   /api/v1/incidents/{id}/acknowledge  Acknowledge an open incident
POST   /api/v1/incidents/{id}/close        Close an incident
POST   /api/v1/machines/{id}/maintenance   Toggle maintenance mode
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import ValidationError as PydanticValidationError

from faultscope.alerting.api.schemas import (
    AcknowledgeRequest,
    CloseRequest,
    EvaluatePredictionRequest,
    IncidentListResponse,
    IncidentResponse,
    MaintenanceModeRequest,
)
from faultscope.alerting.coordinator import IncidentCoordinator
from faultscope.common.exceptions import DatabaseError
from faultscope.common.kafka.schemas import RulPrediction
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/incidents",
    tags=["Incidents"],
)

machines_router = APIRouter(
    prefix="/api/v1/machines",
    tags=["Machines"],
)


def _get_coordinator(request: Request) -> IncidentCoordinator:
    """Extract the ``IncidentCoordinator`` from the app state.

    Parameters
    ----------
    request:
        The current FastAPI request object.

    Returns
    -------
    IncidentCoordinator
        The shared coordinator instance stored on ``app.state``.
    """
    coordinator: IncidentCoordinator = request.app.state.coordinator
    return coordinator


@router.post(
    "/evaluate",
    summary="Submit a prediction for rule evaluation",
    response_model=dict,
)
async def evaluate_prediction(
    body: EvaluatePredictionRequest,
    coordinator: IncidentCoordinator = Depends(_get_coordinator),  # noqa: B008
) -> dict[str, object]:
    """Evaluate a prediction event against all detection rules.

    Returns the list of created incident IDs (may be empty when no
    rules fire for the given prediction values).

    Parameters
    ----------
    body:
        Prediction data to evaluate.

    Returns
    -------
    dict[str, object]
        ``{"machine_id": ..., "incidents_created": N, "incident_ids": [...]}``.
    """
    try:
        prediction = RulPrediction(
            machine_id=body.machine_id,
            predicted_at=body.predicted_at,
            rul_cycles=body.rul_cycles,
            rul_hours=body.rul_hours,
            rul_lower_bound=0.0,
            rul_upper_bound=body.rul_cycles * 2,
            health_label=body.health_label,  # type: ignore[arg-type]
            anomaly_score=body.anomaly_score,
            confidence=body.confidence,
            rul_model_version=body.rul_model_version,
            health_model_version=body.health_model_version,
        )
    except PydanticValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(),
        ) from exc

    try:
        incident_ids = await coordinator.process_prediction(prediction)
    except DatabaseError as exc:
        _log.error(
            "evaluate_prediction_db_error",
            machine_id=body.machine_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=503,
            detail="Database error while processing prediction.",
        ) from exc

    return {
        "machine_id": body.machine_id,
        "incidents_created": len(incident_ids),
        "incident_ids": incident_ids,
    }


@router.get(
    "",
    summary="List and filter incidents",
    response_model=IncidentListResponse,
)
async def list_incidents(
    machine_id: str | None = Query(None, description="Filter by machine ID"),
    status: str | None = Query(
        None,
        description="Filter by status: open, acknowledged, closed",
    ),
    severity: str | None = Query(
        None,
        description="Filter by severity: info, warning, critical",
    ),
    limit: int = Query(50, le=200, ge=1),
    offset: int = Query(0, ge=0),
    coordinator: IncidentCoordinator = Depends(_get_coordinator),  # noqa: B008
) -> IncidentListResponse:
    """Return a paginated list of incidents with optional filters.

    Parameters
    ----------
    machine_id:
        Exact-match filter on machine identifier.
    status:
        Filter by lifecycle status.
    severity:
        Filter by severity label.
    limit:
        Maximum incidents to return (1–200).
    offset:
        Pagination offset.

    Returns
    -------
    IncidentListResponse
        Paginated incidents with ``total`` and ``has_more`` fields.
    """
    try:
        rows, total = await coordinator.list_incidents(
            machine_id=machine_id,
            status=status,
            severity=severity,
            limit=limit,
            offset=offset,
        )
    except DatabaseError as exc:
        _log.error("list_incidents_db_error", error=str(exc))
        raise HTTPException(
            status_code=503,
            detail="Database error while listing incidents.",
        ) from exc

    incidents = [
        IncidentResponse(
            incident_id=str(row["incident_id"]),
            rule_id=str(row["rule_id"]),
            machine_id=str(row["machine_id"]),
            severity=str(row["severity"]),
            title=str(row["title"]),
            status=str(row["status"]),
            triggered_at=_ensure_tz(row["triggered_at"]),
            acknowledged_at=_opt_ensure_tz(row.get("acknowledged_at")),
            closed_at=_opt_ensure_tz(row.get("closed_at")),
        )
        for row in rows
    ]

    return IncidentListResponse(
        total=total,
        incidents=incidents,
        has_more=(offset + len(incidents)) < total,
    )


@router.post(
    "/{incident_id}/acknowledge",
    summary="Acknowledge an open incident",
)
async def acknowledge_incident(
    incident_id: str,
    body: AcknowledgeRequest,
    coordinator: IncidentCoordinator = Depends(_get_coordinator),  # noqa: B008
) -> dict[str, str]:
    """Transition an incident from ``open`` to ``acknowledged``.

    Parameters
    ----------
    incident_id:
        UUID of the target incident.
    body:
        Contains ``acknowledged_by`` field.

    Returns
    -------
    dict[str, str]
        ``{"status": "acknowledged", "incident_id": "..."}``.
    """
    try:
        await coordinator.acknowledge_incident(
            incident_id=incident_id,
            acknowledged_by=body.acknowledged_by,
        )
    except DatabaseError as exc:
        _log.error(
            "acknowledge_incident_db_error",
            incident_id=incident_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=503,
            detail="Database error while acknowledging incident.",
        ) from exc

    return {
        "status": "acknowledged",
        "incident_id": incident_id,
    }


@router.post(
    "/{incident_id}/close",
    summary="Close an incident",
)
async def close_incident(
    incident_id: str,
    body: CloseRequest,
    coordinator: IncidentCoordinator = Depends(_get_coordinator),  # noqa: B008
) -> dict[str, str]:
    """Transition an incident to ``closed``.

    Parameters
    ----------
    incident_id:
        UUID of the target incident.
    body:
        Optional ``resolution_note`` describing the resolution.

    Returns
    -------
    dict[str, str]
        ``{"status": "closed", "incident_id": "..."}``.
    """
    try:
        await coordinator.close_incident(
            incident_id=incident_id,
            resolution_note=body.resolution_note,
        )
    except DatabaseError as exc:
        _log.error(
            "close_incident_db_error",
            incident_id=incident_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=503,
            detail="Database error while closing incident.",
        ) from exc

    return {
        "status": "closed",
        "incident_id": incident_id,
    }


@machines_router.post(
    "/{machine_id}/maintenance",
    summary="Toggle maintenance mode for a machine",
    tags=["Machines"],
)
async def set_maintenance_mode(
    machine_id: str,
    body: MaintenanceModeRequest,
    coordinator: IncidentCoordinator = Depends(_get_coordinator),  # noqa: B008
) -> dict[str, object]:
    """Enable or disable alert suppression for a machine in maintenance.

    Parameters
    ----------
    machine_id:
        Target machine identifier.
    body:
        ``{"enabled": true}`` to suppress, ``{"enabled": false}``
        to resume normal alerting.

    Returns
    -------
    dict[str, object]
        ``{"machine_id": ..., "maintenance_mode": true/false}``.
    """
    coordinator.set_maintenance_mode(machine_id, body.enabled)
    _log.info(
        "maintenance_mode_updated",
        machine_id=machine_id,
        enabled=body.enabled,
    )
    return {
        "machine_id": machine_id,
        "maintenance_mode": body.enabled,
    }


# ------------------------------------------------------------------ #
# Datetime helpers
# ------------------------------------------------------------------ #


def _ensure_tz(dt: object) -> datetime:
    """Ensure a datetime object is timezone-aware (UTC).

    Parameters
    ----------
    dt:
        Value retrieved from the database row.

    Returns
    -------
    datetime
        Timezone-aware UTC datetime.
    """
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt
    raise TypeError(f"Expected datetime, got {type(dt)}")


def _opt_ensure_tz(
    dt: object,
) -> datetime | None:
    """Return a timezone-aware datetime or ``None``.

    Parameters
    ----------
    dt:
        Value that may be ``None`` or a naive/aware datetime.

    Returns
    -------
    datetime | None
        ``None`` if ``dt`` is ``None``, otherwise a UTC datetime.
    """
    if dt is None:
        return None
    return _ensure_tz(dt)
