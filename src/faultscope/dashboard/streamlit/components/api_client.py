"""Thin synchronous HTTP client for the FaultScope APIs.

All dashboard pages should obtain a client via ``get_inference_client``
or ``get_alerting_client`` and call the typed helper methods rather
than issuing raw ``httpx`` calls.  Errors are surfaced as structured
log entries; callers receive ``None`` (or an empty container) so the
UI can display a graceful error message instead of crashing.
"""

from __future__ import annotations

import httpx
import structlog

from faultscope.dashboard.streamlit.config import DashboardConfig

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Default request timeout (seconds)
_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0)


def _auth_headers(cfg: DashboardConfig) -> dict[str, str]:
    """Return ``Authorization`` header dict if an API key is set."""
    if cfg.api_key:
        return {"Authorization": f"Bearer {cfg.api_key}"}
    return {}


# ── Inference API helpers


def fetch_machines(cfg: DashboardConfig) -> list[dict[str, object]]:
    """GET /api/v1/machines — list all registered machines.

    Returns
    -------
    list[dict[str, object]]
        Parsed JSON list, or ``[]`` on error.
    """
    url = f"{cfg.inference_base_url}/api/v1/machines"
    try:
        resp = httpx.get(url, headers=_auth_headers(cfg), timeout=_TIMEOUT)
        resp.raise_for_status()
        data: list[dict[str, object]] = resp.json()
        return data
    except httpx.HTTPError as exc:
        log.warning(
            "fetch_machines_failed",
            url=url,
            error=str(exc),
        )
        return []


def fetch_latest_predictions(
    cfg: DashboardConfig,
) -> list[dict[str, object]]:
    """GET /api/v1/predictions — latest RUL predictions for all machines.

    Returns
    -------
    list[dict[str, object]]
        Parsed JSON list, or ``[]`` on error.
    """
    url = f"{cfg.inference_base_url}/api/v1/predictions"
    try:
        resp = httpx.get(url, headers=_auth_headers(cfg), timeout=_TIMEOUT)
        resp.raise_for_status()
        data: list[dict[str, object]] = resp.json()
        return data
    except httpx.HTTPError as exc:
        log.warning(
            "fetch_latest_predictions_failed",
            url=url,
            error=str(exc),
        )
        return []


def fetch_machine_predictions(
    cfg: DashboardConfig,
    machine_id: str,
    hours: int = 24,
) -> list[dict[str, object]]:
    """GET /api/v1/predictions/{machine_id} — time-series for one machine.

    Parameters
    ----------
    cfg:
        Dashboard configuration.
    machine_id:
        Target machine identifier.
    hours:
        How many hours of history to retrieve.

    Returns
    -------
    list[dict[str, object]]
        Ordered list of prediction dicts (oldest first), or ``[]``.
    """
    url = (
        f"{cfg.inference_base_url}"
        f"/api/v1/predictions/{machine_id}"
        f"?hours={hours}"
    )
    try:
        resp = httpx.get(url, headers=_auth_headers(cfg), timeout=_TIMEOUT)
        resp.raise_for_status()
        data: list[dict[str, object]] = resp.json()
        return data
    except httpx.HTTPError as exc:
        log.warning(
            "fetch_machine_predictions_failed",
            url=url,
            machine_id=machine_id,
            error=str(exc),
        )
        return []


def fetch_sensor_readings(
    cfg: DashboardConfig,
    machine_id: str,
    hours: int = 1,
) -> list[dict[str, object]]:
    """GET /api/v1/sensors/{machine_id} — recent sensor readings.

    Parameters
    ----------
    cfg:
        Dashboard configuration.
    machine_id:
        Target machine identifier.
    hours:
        How many hours of history to retrieve.

    Returns
    -------
    list[dict[str, object]]
        Ordered list of reading dicts, or ``[]``.
    """
    url = f"{cfg.inference_base_url}/api/v1/sensors/{machine_id}?hours={hours}"
    try:
        resp = httpx.get(url, headers=_auth_headers(cfg), timeout=_TIMEOUT)
        resp.raise_for_status()
        data: list[dict[str, object]] = resp.json()
        return data
    except httpx.HTTPError as exc:
        log.warning(
            "fetch_sensor_readings_failed",
            url=url,
            machine_id=machine_id,
            error=str(exc),
        )
        return []


def fetch_inference_models(
    cfg: DashboardConfig,
) -> list[dict[str, object]]:
    """GET /api/v1/models — loaded model versions and metadata.

    Returns
    -------
    list[dict[str, object]]
        Parsed JSON list, or ``[]`` on error.
    """
    url = f"{cfg.inference_base_url}/api/v1/models"
    try:
        resp = httpx.get(url, headers=_auth_headers(cfg), timeout=_TIMEOUT)
        resp.raise_for_status()
        data: list[dict[str, object]] = resp.json()
        return data
    except httpx.HTTPError as exc:
        log.warning(
            "fetch_inference_models_failed",
            url=url,
            error=str(exc),
        )
        return []


def fetch_inference_health(
    cfg: DashboardConfig,
) -> dict[str, object]:
    """GET /health — inference service health and Kafka consumer lag.

    Returns
    -------
    dict[str, object]
        Health payload, or ``{}`` on error.
    """
    url = f"{cfg.inference_base_url}/health"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        return data
    except httpx.HTTPError as exc:
        log.warning(
            "fetch_inference_health_failed",
            url=url,
            error=str(exc),
        )
        return {}


def fetch_recent_predictions_sample(
    cfg: DashboardConfig,
    limit: int = 100,
) -> list[dict[str, object]]:
    """GET /api/v1/predictions?limit=N — last N predictions (all machines).

    Returns
    -------
    list[dict[str, object]]
        Parsed JSON list, or ``[]`` on error.
    """
    url = f"{cfg.inference_base_url}/api/v1/predictions?limit={limit}"
    try:
        resp = httpx.get(url, headers=_auth_headers(cfg), timeout=_TIMEOUT)
        resp.raise_for_status()
        data: list[dict[str, object]] = resp.json()
        return data
    except httpx.HTTPError as exc:
        log.warning(
            "fetch_recent_predictions_sample_failed",
            url=url,
            error=str(exc),
        )
        return []


# ── Alerting API helpers


def fetch_incidents(
    cfg: DashboardConfig,
    machine_id: str | None = None,
    status: str | None = None,
    severity: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, object]:
    """GET /api/v1/incidents — paginated incident list.

    Parameters
    ----------
    cfg:
        Dashboard configuration.
    machine_id:
        Optional filter by machine identifier.
    status:
        Optional filter: ``"open"``, ``"acknowledged"``, ``"closed"``.
    severity:
        Optional filter: ``"info"``, ``"warning"``, ``"critical"``.
    page:
        1-based page index.
    page_size:
        Results per page.

    Returns
    -------
    dict[str, object]
        Parsed JSON with ``items``, ``total``, ``page``, ``pages``,
        or ``{"items": [], "total": 0, "page": 1, "pages": 0}`` on error.
    """
    params: dict[str, str | int] = {
        "page": page,
        "page_size": page_size,
    }
    if machine_id:
        params["machine_id"] = machine_id
    if status:
        params["status"] = status
    if severity:
        params["severity"] = severity

    url = f"{cfg.alerting_base_url}/api/v1/incidents"
    try:
        resp = httpx.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        return data
    except httpx.HTTPError as exc:
        log.warning(
            "fetch_incidents_failed",
            url=url,
            error=str(exc),
        )
        return {"items": [], "total": 0, "page": 1, "pages": 0}


def fetch_active_incidents(
    cfg: DashboardConfig,
) -> list[dict[str, object]]:
    """GET /api/v1/incidents?status=open — all open incidents.

    Returns
    -------
    list[dict[str, object]]
        Parsed JSON list, or ``[]`` on error.
    """
    result = fetch_incidents(cfg, status="open", page=1, page_size=200)
    items = result.get("items", [])
    return items if isinstance(items, list) else []


def acknowledge_incident(
    cfg: DashboardConfig,
    incident_id: str,
) -> bool:
    """POST /api/v1/incidents/{id}/acknowledge — acknowledge an incident.

    Parameters
    ----------
    cfg:
        Dashboard configuration.
    incident_id:
        UUID of the incident to acknowledge.

    Returns
    -------
    bool
        ``True`` if successful.
    """
    url = f"{cfg.alerting_base_url}/api/v1/incidents/{incident_id}/acknowledge"
    try:
        resp = httpx.post(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        log.warning(
            "acknowledge_incident_failed",
            url=url,
            incident_id=incident_id,
            error=str(exc),
        )
        return False


def close_incident(
    cfg: DashboardConfig,
    incident_id: str,
) -> bool:
    """POST /api/v1/incidents/{id}/close — close an incident.

    Parameters
    ----------
    cfg:
        Dashboard configuration.
    incident_id:
        UUID of the incident to close.

    Returns
    -------
    bool
        ``True`` if successful.
    """
    url = f"{cfg.alerting_base_url}/api/v1/incidents/{incident_id}/close"
    try:
        resp = httpx.post(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return True
    except httpx.HTTPError as exc:
        log.warning(
            "close_incident_failed",
            url=url,
            incident_id=incident_id,
            error=str(exc),
        )
        return False


def fetch_alerting_health(
    cfg: DashboardConfig,
) -> dict[str, object]:
    """GET /health — alerting service health payload.

    Returns
    -------
    dict[str, object]
        Health payload, or ``{}`` on error.
    """
    url = f"{cfg.alerting_base_url}/health"
    try:
        resp = httpx.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data: dict[str, object] = resp.json()
        return data
    except httpx.HTTPError as exc:
        log.warning(
            "fetch_alerting_health_failed",
            url=url,
            error=str(exc),
        )
        return {}
