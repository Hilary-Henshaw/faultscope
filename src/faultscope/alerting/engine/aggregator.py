"""Alert aggregation and deduplication for the alerting service.

``IncidentAggregator`` groups triggered incidents that arrive within a
configurable time window for the same machine.  Within the window, only
the highest-severity incident per ``rule_id`` is retained.  When
``flush()`` is called, all pending incident groups are returned and the
internal buffer is cleared.

This reduces notification noise by batching related alerts that occur
close together in time (e.g. a prediction storm after a sensor reconnect).

Usage::

    agg = IncidentAggregator(aggregation_window_s=300)
    agg.add(incident)
    ...
    groups = agg.flush()  # called every aggregation_window_s seconds
    for group in groups:
        await dispatch(group)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from faultscope.alerting.engine.evaluator import TriggeredIncident
from faultscope.alerting.rules import Severity
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

# Ordered severity levels for comparison — higher index = higher severity.
_SEVERITY_ORDER: dict[Severity, int] = {
    Severity.INFO: 0,
    Severity.WARNING: 1,
    Severity.CRITICAL: 2,
}


@dataclass
class _WindowBucket:
    """Internal state for a single (machine_id, rule_id) aggregation slot.

    Attributes
    ----------
    best:
        The highest-severity incident seen in the current window.
    window_start:
        UTC timestamp when the first incident landed in this slot.
    """

    best: TriggeredIncident
    window_start: datetime = field(
        default_factory=lambda: datetime.now(tz=UTC)
    )


class IncidentAggregator:
    """Group and deduplicate incidents within a rolling time window.

    Within ``aggregation_window_s`` for the same ``machine_id``:

    - Only the highest-severity incident per ``rule_id`` is retained.
    - Duplicate incidents (same machine + rule) are suppressed.
    - ``flush()`` drains all groups and resets the buffer.

    Parameters
    ----------
    aggregation_window_s:
        Duration in seconds that defines the aggregation window.
        Incidents arriving within this window for the same machine are
        candidates for deduplication.
    """

    def __init__(self, aggregation_window_s: int = 300) -> None:
        self._window_s: int = aggregation_window_s
        # (machine_id, rule_id) → _WindowBucket
        self._buckets: dict[tuple[str, str], _WindowBucket] = {}

        _log.debug(
            "incident_aggregator_initialized",
            aggregation_window_s=aggregation_window_s,
        )

    def add(self, incident: TriggeredIncident) -> None:
        """Add an incident to the aggregation buffer.

        If a bucket already exists for (machine_id, rule_id) and is
        still within the aggregation window, the stored incident is
        replaced only when the new one has higher or equal severity.
        Incidents that arrive after the window expires for their slot
        start a fresh bucket.

        Parameters
        ----------
        incident:
            The triggered incident to buffer.
        """
        now = datetime.now(tz=UTC)
        key = (incident.machine_id, incident.rule.rule_id)

        existing = self._buckets.get(key)

        if existing is not None:
            age_s = (now - existing.window_start).total_seconds()
            if age_s <= self._window_s:
                # Deduplicate: keep the highest severity.
                if (
                    _SEVERITY_ORDER[incident.severity]
                    >= _SEVERITY_ORDER[existing.best.severity]
                ):
                    existing.best = incident
                    _log.debug(
                        "incident_aggregated_upgraded",
                        machine_id=incident.machine_id,
                        rule_id=incident.rule.rule_id,
                        severity=incident.severity.value,
                    )
                else:
                    _log.debug(
                        "incident_aggregated_suppressed",
                        machine_id=incident.machine_id,
                        rule_id=incident.rule.rule_id,
                        existing_severity=existing.best.severity.value,
                        new_severity=incident.severity.value,
                    )
                return
            # Window expired — start a fresh bucket for this slot.

        self._buckets[key] = _WindowBucket(best=incident, window_start=now)
        _log.debug(
            "incident_aggregated_new_bucket",
            machine_id=incident.machine_id,
            rule_id=incident.rule.rule_id,
            severity=incident.severity.value,
        )

    def flush(self) -> list[list[TriggeredIncident]]:
        """Return all buffered incident groups and reset the buffer.

        Incidents are grouped by ``machine_id``.  All buckets — even
        those still inside their aggregation window — are flushed and
        the buffer is cleared.  The caller is responsible for ensuring
        this is invoked on the right schedule (typically every
        ``aggregation_window_s`` seconds).

        Returns
        -------
        list[list[TriggeredIncident]]
            Each inner list contains all deduplicated incidents for one
            machine, sorted by severity (highest first).  Empty when no
            incidents have been buffered since the last flush.
        """
        if not self._buckets:
            return []

        by_machine: dict[str, list[TriggeredIncident]] = defaultdict(list)
        for bucket in self._buckets.values():
            by_machine[bucket.best.machine_id].append(bucket.best)

        # Sort each machine's group: CRITICAL first, then WARNING, INFO.
        groups: list[list[TriggeredIncident]] = []
        for incidents in by_machine.values():
            incidents.sort(
                key=lambda i: _SEVERITY_ORDER[i.severity],
                reverse=True,
            )
            groups.append(incidents)

        total = sum(len(g) for g in groups)
        _log.info(
            "incident_aggregator_flushed",
            machine_groups=len(groups),
            total_incidents=total,
        )

        self._buckets.clear()
        return groups
