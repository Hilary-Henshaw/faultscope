"""Pydantic request/response schemas for the alerting service REST API.

All models use Pydantic v2.  Datetime fields are serialised as ISO-8601
UTC strings consistent with the rest of the FaultScope pipeline.

Usage::

    from faultscope.alerting.api.schemas import (
        EvaluatePredictionRequest,
        IncidentResponse,
    )
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class EvaluatePredictionRequest(BaseModel):
    """Request body for ``POST /api/v1/incidents/evaluate``.

    Attributes
    ----------
    machine_id:
        Unique identifier of the machine being evaluated.
    rul_cycles:
        Remaining useful life in engine cycles.
    rul_hours:
        Remaining useful life in hours.
    anomaly_score:
        Anomaly score in ``[0, 1]``.
    health_label:
        Discrete health classification string.
    confidence:
        Model confidence in ``[0, 1]``.
    rul_model_version:
        Semantic version tag of the RUL model.
    health_model_version:
        Semantic version tag of the health model.
    predicted_at:
        Timestamp of the prediction; defaults to ``utcnow``.
    """

    model_config = ConfigDict(populate_by_name=True)

    machine_id: str = Field(..., min_length=1, max_length=64)
    rul_cycles: float = Field(..., ge=0.0)
    rul_hours: float = Field(..., ge=0.0)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    health_label: str = Field(..., min_length=1, max_length=32)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rul_model_version: str = Field(default="unknown")
    health_model_version: str = Field(default="unknown")
    predicted_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=UTC)
    )


class IncidentResponse(BaseModel):
    """Single incident serialised in API responses.

    Attributes
    ----------
    incident_id:
        UUID string.
    rule_id:
        Identifier of the detection rule that triggered this incident.
    machine_id:
        Machine that generated the incident.
    severity:
        Severity level string.
    title:
        Short human-readable title.
    status:
        Lifecycle status: ``open``, ``acknowledged``, or ``closed``.
    triggered_at:
        UTC timestamp when the incident was first triggered.
    acknowledged_at:
        UTC timestamp of acknowledgement; ``None`` if not acknowledged.
    closed_at:
        UTC timestamp of closure; ``None`` if not closed.
    """

    model_config = ConfigDict(populate_by_name=True)

    incident_id: str
    rule_id: str
    machine_id: str
    severity: str
    title: str
    status: str
    triggered_at: datetime
    acknowledged_at: datetime | None = None
    closed_at: datetime | None = None


class IncidentListResponse(BaseModel):
    """Paginated list of incidents.

    Attributes
    ----------
    total:
        Total number of incidents matching the applied filters (not
        just the current page).
    incidents:
        The incidents on the current page.
    has_more:
        ``True`` when there are further pages beyond this response.
    """

    model_config = ConfigDict(populate_by_name=True)

    total: int
    incidents: list[IncidentResponse]
    has_more: bool


class AcknowledgeRequest(BaseModel):
    """Request body for ``POST /api/v1/incidents/{id}/acknowledge``.

    Attributes
    ----------
    acknowledged_by:
        Identity of the person acknowledging (email, username, etc.).
    """

    model_config = ConfigDict(populate_by_name=True)

    acknowledged_by: str = Field(..., min_length=1, max_length=100)


class CloseRequest(BaseModel):
    """Request body for ``POST /api/v1/incidents/{id}/close``.

    Attributes
    ----------
    resolution_note:
        Optional free-text description of the resolution action taken.
    """

    model_config = ConfigDict(populate_by_name=True)

    resolution_note: str = Field(default="", max_length=1000)


class MaintenanceModeRequest(BaseModel):
    """Request body for ``POST /api/v1/machines/{id}/maintenance``.

    Attributes
    ----------
    enabled:
        ``True`` to activate maintenance-mode suppression for the
        machine, ``False`` to deactivate it.
    """

    model_config = ConfigDict(populate_by_name=True)

    enabled: bool
