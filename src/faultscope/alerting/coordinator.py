"""Central alert coordinator for the FaultScope alerting service.

``IncidentCoordinator`` is the single orchestration point for
prediction-to-notification flow.  It wires together:

- ``RuleEvaluationEngine`` – evaluates all detection rules
- ``IncidentAggregator`` – deduplicates and groups incidents
- ``IncidentSuppressor`` – silences maintenance / quiet-hours alerts
- Configured notifiers – dispatches grouped notifications
- ``asyncpg.Pool`` – persists incidents to the ``alerting.incidents``
  table

Incident lifecycle
------------------
``open`` → (acknowledged_by someone) → ``acknowledged``
         → (resolved with a note)     → ``closed``

Usage::

    coordinator = IncidentCoordinator(config, pool, notifiers)
    incident_ids = await coordinator.process_prediction(prediction)
"""

from __future__ import annotations

import json
import uuid

import asyncpg

from faultscope.alerting.config import AlertingConfig
from faultscope.alerting.engine.aggregator import IncidentAggregator
from faultscope.alerting.engine.evaluator import (
    RuleEvaluationEngine,
    TriggeredIncident,
)
from faultscope.alerting.engine.suppressor import IncidentSuppressor
from faultscope.alerting.notifiers.base import (
    BaseNotifier,
    NotificationPayload,
)
from faultscope.alerting.rules import (
    DEFAULT_RULES,
    EvaluationContext,
    Severity,
)
from faultscope.common.exceptions import DatabaseError
from faultscope.common.kafka.schemas import RulPrediction
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

# Ordered severity for picking the "max" across a group.
_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.CRITICAL: 2,
}

# SQL templates

_INSERT_INCIDENT = """
INSERT INTO alerting.incidents (
    incident_id, rule_id, machine_id, severity, title,
    status, details, triggered_at, created_at
) VALUES (
    $1, $2, $3, $4, $5,
    'open', $6, $7, NOW()
)
"""

_FETCH_INCIDENT = """
SELECT incident_id, rule_id, machine_id, severity, title,
       status, triggered_at, acknowledged_at, closed_at, details
FROM alerting.incidents
WHERE incident_id = $1
"""

_ACK_INCIDENT = """
UPDATE alerting.incidents
SET status = 'acknowledged',
    acknowledged_at = NOW(),
    acknowledged_by = $2
WHERE incident_id = $1
  AND status = 'open'
"""

_CLOSE_INCIDENT = """
UPDATE alerting.incidents
SET status = 'closed',
    closed_at = NOW(),
    resolution_note = $2
WHERE incident_id = $1
  AND status IN ('open', 'acknowledged')
"""

_LIST_INCIDENTS_BASE = """
SELECT incident_id, rule_id, machine_id, severity, title,
       status, triggered_at, acknowledged_at, closed_at
FROM alerting.incidents
WHERE 1=1
"""

_COUNT_INCIDENTS_BASE = """
SELECT COUNT(*) FROM alerting.incidents WHERE 1=1
"""

_LATEST_RUL_FOR_MACHINE = """
SELECT rul_cycles
FROM alerting.incidents
WHERE machine_id = $1
ORDER BY triggered_at DESC
LIMIT 1
"""


class IncidentCoordinator:
    """Orchestrate prediction evaluation, deduplication, and notification.

    Parameters
    ----------
    config:
        Resolved ``AlertingConfig`` instance.
    db_pool:
        Async ``asyncpg`` connection pool connected to the FaultScope
        database.
    notifiers:
        All active notification channel implementations.
    """

    def __init__(
        self,
        config: AlertingConfig,
        db_pool: asyncpg.Pool,  # type: ignore[type-arg]
        notifiers: list[BaseNotifier],
    ) -> None:
        self._config = config
        self._pool = db_pool
        self._notifiers = notifiers

        self._engine = RuleEvaluationEngine(DEFAULT_RULES)
        self._aggregator = IncidentAggregator(config.aggregation_window_s)
        self._suppressor = IncidentSuppressor()

        # Per-machine last known RUL for rate-of-change rules.
        self._last_rul: dict[str, float] = {}

        _log.info(
            "incident_coordinator_initialized",
            notifiers=[n.channel_name for n in notifiers],
            aggregation_window_s=config.aggregation_window_s,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def process_prediction(
        self,
        prediction: RulPrediction,
    ) -> list[str]:
        """Evaluate detection rules and dispatch notifications.

        Steps:
        1. Build ``EvaluationContext`` from in-memory state.
        2. Run ``RuleEvaluationEngine.evaluate``.
        3. Persist each triggered incident to the database.
        4. Feed incidents into ``IncidentAggregator``.
        5. Flush aggregator; for each group, check suppressor, then
           dispatch to all notifiers.
        6. Update per-machine last-RUL cache.

        Parameters
        ----------
        prediction:
            The incoming RUL prediction event.

        Returns
        -------
        list[str]
            UUIDs of all persisted incidents (may be empty).
        """
        machine_id = prediction.machine_id
        context = EvaluationContext(
            previous_rul=self._last_rul.get(machine_id),
            active_sensor_count=self._count_anomalous_sensors(prediction),
            machine_in_maintenance=machine_id
            in self._suppressor.maintenance_machines,
        )

        incidents = self._engine.evaluate(prediction, context)

        incident_ids: list[str] = []
        for inc in incidents:
            iid = await self._persist_incident(inc, prediction)
            incident_ids.append(iid)
            self._aggregator.add(inc)

        groups = self._aggregator.flush()
        for group in groups:
            await self._dispatch_group(group)

        # Update last-known RUL for next prediction.
        self._last_rul[machine_id] = prediction.rul_cycles

        if incident_ids:
            _log.info(
                "prediction_processed",
                machine_id=machine_id,
                incidents_created=len(incident_ids),
                incident_ids=incident_ids,
            )
        else:
            _log.debug(
                "prediction_processed_no_incidents",
                machine_id=machine_id,
                rul_cycles=prediction.rul_cycles,
            )

        return incident_ids

    async def acknowledge_incident(
        self,
        incident_id: str,
        acknowledged_by: str,
    ) -> None:
        """Transition an incident from ``open`` to ``acknowledged``.

        Parameters
        ----------
        incident_id:
            UUID string of the incident to acknowledge.
        acknowledged_by:
            Identity of the person performing the acknowledgement.

        Raises
        ------
        DatabaseError
            If the database update fails.
        """
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    _ACK_INCIDENT, incident_id, acknowledged_by
                )
        except asyncpg.PostgresError as exc:
            raise DatabaseError(
                f"Failed to acknowledge incident {incident_id}: {exc}",
                context={
                    "incident_id": incident_id,
                    "error": str(exc),
                },
            ) from exc

        _log.info(
            "incident_acknowledged",
            incident_id=incident_id,
            acknowledged_by=acknowledged_by,
            db_result=result,
        )

    async def close_incident(
        self,
        incident_id: str,
        resolution_note: str = "",
    ) -> None:
        """Transition an incident to ``closed``.

        Parameters
        ----------
        incident_id:
            UUID string of the incident to close.
        resolution_note:
            Optional free-text description of how the issue was
            resolved.

        Raises
        ------
        DatabaseError
            If the database update fails.
        """
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    _CLOSE_INCIDENT, incident_id, resolution_note
                )
        except asyncpg.PostgresError as exc:
            raise DatabaseError(
                f"Failed to close incident {incident_id}: {exc}",
                context={
                    "incident_id": incident_id,
                    "error": str(exc),
                },
            ) from exc

        _log.info(
            "incident_closed",
            incident_id=incident_id,
            db_result=result,
        )

    async def list_incidents(
        self,
        machine_id: str | None = None,
        status: str | None = None,
        severity: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, object]], int]:
        """Return paginated incidents matching the supplied filters.

        Parameters
        ----------
        machine_id:
            Filter by machine identifier (exact match).
        status:
            Filter by lifecycle status (``open``, ``acknowledged``,
            ``closed``).
        severity:
            Filter by severity label (``info``, ``warning``,
            ``critical``).
        limit:
            Maximum number of rows to return.
        offset:
            Row offset for pagination.

        Returns
        -------
        tuple[list[dict[str, object]], int]
            ``(rows, total_count)`` where ``rows`` is a list of
            incident dicts and ``total_count`` is the unfiltered
            matching count.

        Raises
        ------
        DatabaseError
            If the database query fails.
        """
        conditions: list[str] = []
        args: list[object] = []
        idx = 1

        if machine_id is not None:
            conditions.append(f"machine_id = ${idx}")
            args.append(machine_id)
            idx += 1
        if status is not None:
            conditions.append(f"status = ${idx}")
            args.append(status)
            idx += 1
        if severity is not None:
            conditions.append(f"severity = ${idx}")
            args.append(severity)
            idx += 1

        where_clause = (
            " AND ".join(f" AND {c}" for c in conditions) if conditions else ""
        )

        count_query = _COUNT_INCIDENTS_BASE + where_clause
        list_query = (
            _LIST_INCIDENTS_BASE
            + where_clause
            + f" ORDER BY triggered_at DESC"
            f" LIMIT ${idx} OFFSET ${idx + 1}"
        )

        try:
            async with self._pool.acquire() as conn:
                total: int = await conn.fetchval(count_query, *args)
                rows = await conn.fetch(list_query, *args, limit, offset)
        except asyncpg.PostgresError as exc:
            raise DatabaseError(
                f"Failed to list incidents: {exc}",
                context={"error": str(exc)},
            ) from exc

        result: list[dict[str, object]] = [dict(row) for row in rows]
        return result, int(total)

    def set_maintenance_mode(self, machine_id: str, enabled: bool) -> None:
        """Proxy maintenance-mode changes to the suppressor.

        Parameters
        ----------
        machine_id:
            Target machine identifier.
        enabled:
            ``True`` to activate suppression, ``False`` to lift it.
        """
        self._suppressor.set_maintenance_mode(machine_id, enabled)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    async def _persist_incident(
        self,
        incident: TriggeredIncident,
        prediction: RulPrediction,
    ) -> str:
        """Write a triggered incident to the database.

        Parameters
        ----------
        incident:
            The triggered incident to persist.
        prediction:
            Original prediction for additional context stored in
            ``details``.

        Returns
        -------
        str
            The generated UUID for the persisted incident.

        Raises
        ------
        DatabaseError
            If the INSERT fails.
        """
        incident_id = str(uuid.uuid4())
        details = dict(incident.details)
        details["machine_id"] = prediction.machine_id
        details_json = json.dumps(details, default=str)

        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    _INSERT_INCIDENT,
                    incident_id,
                    incident.rule.rule_id,
                    incident.machine_id,
                    incident.severity.value,
                    incident.title,
                    details_json,
                    incident.triggered_at,
                )
        except asyncpg.PostgresError as exc:
            raise DatabaseError(
                f"Failed to persist incident for machine "
                f"{incident.machine_id}: {exc}",
                context={
                    "machine_id": incident.machine_id,
                    "rule_id": incident.rule.rule_id,
                    "error": str(exc),
                },
            ) from exc

        _log.debug(
            "incident_persisted",
            incident_id=incident_id,
            machine_id=incident.machine_id,
            rule_id=incident.rule.rule_id,
        )
        return incident_id

    async def _dispatch_group(
        self,
        group: list[TriggeredIncident],
    ) -> None:
        """Build a ``NotificationPayload`` and send to all notifiers.

        The group is suppressed entirely if the suppressor says so for
        the machine + highest severity.

        Parameters
        ----------
        group:
            Deduplicated incidents for one machine, highest severity
            first.
        """
        if not group:
            return

        machine_id = group[0].machine_id
        max_severity = max(
            (i.severity for i in group),
            key=lambda s: _SEVERITY_ORDER[s],
        )

        if self._suppressor.should_suppress(machine_id, max_severity):
            _log.info(
                "incident_group_suppressed",
                machine_id=machine_id,
                severity=max_severity.value,
                incident_count=len(group),
            )
            return

        payload = NotificationPayload(
            machine_id=machine_id,
            severity=max_severity,
            title=group[0].title,
            incidents=group,
            triggered_at=min(i.triggered_at for i in group),
        )

        for notifier in self._notifiers:
            try:
                await notifier.send(payload)
            except Exception as exc:  # noqa: BLE001
                _log.error(
                    "notifier_unexpected_error",
                    channel=notifier.channel_name,
                    machine_id=machine_id,
                    error=str(exc),
                )

    @staticmethod
    def _count_anomalous_sensors(prediction: RulPrediction) -> int:
        """Estimate the number of anomalous sensors from the prediction.

        Uses the anomaly score as a proxy: sensors are counted as
        anomalous when the overall anomaly score is above 0.5.  This is
        a heuristic because the prediction schema does not carry
        per-sensor anomaly flags; the multi-sensor rule requires the
        count to be supplied from a richer source in production (inject
        via EvaluationContext at the call site).

        Parameters
        ----------
        prediction:
            The prediction event.

        Returns
        -------
        int
            Estimated count of anomalous sensors.
        """
        if prediction.anomaly_score >= 0.9:
            return 4
        if prediction.anomaly_score >= 0.7:
            return 3
        if prediction.anomaly_score >= 0.5:
            return 2
        return 0
