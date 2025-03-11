"""Incident suppression logic for the FaultScope alerting service.

``IncidentSuppressor`` prevents notification spam during planned
maintenance windows and user-configured quiet hours.  The two
suppression criteria are independent: either condition alone is
sufficient to suppress a notification.

Usage::

    suppressor = IncidentSuppressor(
        maintenance_machines={"FAN-001"},
        quiet_hours=(22, 6),  # 22:00–06:00 UTC
    )
    suppressor.set_maintenance_mode("FAN-002", enabled=True)

    if not suppressor.should_suppress("FAN-001", Severity.WARNING):
        await notifier.send(payload)
"""

from __future__ import annotations

from datetime import UTC, datetime

from faultscope.alerting.rules import Severity
from faultscope.common.logging import get_logger

_log = get_logger(__name__)


class IncidentSuppressor:
    """Gate notifications for machines in maintenance or quiet hours.

    Suppression criteria (either is sufficient):

    1. **Maintenance mode** – the machine has been marked as under
       maintenance via ``set_maintenance_mode``.  All severities are
       suppressed during maintenance (operators do not need alert noise
       while actively working on equipment).

    2. **Quiet hours** – a UTC hour range ``(start_hour, end_hour)``
       where ``start_hour > end_hour`` indicates an overnight window
       (e.g. ``(22, 6)`` means 22:00 – 06:00 UTC).  Only
       ``Severity.INFO`` and ``Severity.WARNING`` notifications are
       suppressed during quiet hours; ``Severity.CRITICAL`` always
       passes through.

    Parameters
    ----------
    maintenance_machines:
        Initial set of machine IDs currently in maintenance.  May be
        ``None`` (treated as an empty set).
    quiet_hours:
        ``(start_hour, end_hour)`` in 24-hour UTC time.  ``None``
        disables quiet-hours suppression entirely.
    """

    def __init__(
        self,
        maintenance_machines: set[str] | None = None,
        quiet_hours: tuple[int, int] | None = None,
    ) -> None:
        self._maintenance: set[str] = (
            set(maintenance_machines) if maintenance_machines else set()
        )
        self._quiet_hours: tuple[int, int] | None = quiet_hours

        _log.info(
            "incident_suppressor_initialized",
            maintenance_machines=list(self._maintenance),
            quiet_hours=quiet_hours,
        )

    def should_suppress(
        self,
        machine_id: str,
        severity: Severity,
        now: datetime | None = None,
    ) -> bool:
        """Return ``True`` if the notification should be suppressed.

        Parameters
        ----------
        machine_id:
            The machine for which the incident was raised.
        severity:
            Severity of the incident.
        now:
            Current UTC time.  Defaults to ``datetime.now(utc)``; pass
            an explicit value in tests to control the clock.

        Returns
        -------
        bool
            ``True`` means the notification should be silenced.
        """
        if machine_id in self._maintenance:
            _log.debug(
                "incident_suppressed_maintenance",
                machine_id=machine_id,
                severity=severity.value,
            )
            return True

        if self._quiet_hours is not None:
            effective_now = now or datetime.now(tz=UTC)
            if self._in_quiet_hours(effective_now) and (
                severity is not Severity.CRITICAL
            ):
                _log.debug(
                    "incident_suppressed_quiet_hours",
                    machine_id=machine_id,
                    severity=severity.value,
                    hour=effective_now.hour,
                )
                return True

        return False

    def set_maintenance_mode(
        self,
        machine_id: str,
        enabled: bool,
    ) -> None:
        """Add or remove a machine from the maintenance set.

        Parameters
        ----------
        machine_id:
            Target machine identifier.
        enabled:
            ``True`` to start suppressing alerts for this machine,
            ``False`` to resume normal alerting.
        """
        if enabled:
            self._maintenance.add(machine_id)
            _log.info(
                "maintenance_mode_enabled",
                machine_id=machine_id,
            )
        else:
            self._maintenance.discard(machine_id)
            _log.info(
                "maintenance_mode_disabled",
                machine_id=machine_id,
            )

    @property
    def maintenance_machines(self) -> frozenset[str]:
        """Return an immutable snapshot of maintenance machine IDs.

        Returns
        -------
        frozenset[str]
            Current set of machines in maintenance mode.
        """
        return frozenset(self._maintenance)

    # ------------------------------------------------------------------ #
    # Quiet-hours helper
    # ------------------------------------------------------------------ #

    def _in_quiet_hours(self, now: datetime) -> bool:
        """Return ``True`` when ``now`` falls inside the quiet window.

        Handles both same-day ranges (e.g. 08–18) and overnight ranges
        that span midnight (e.g. 22–06).

        Parameters
        ----------
        now:
            The current UTC datetime to test.

        Returns
        -------
        bool
            ``True`` if quiet hours are active at ``now``.
        """
        assert self._quiet_hours is not None  # guarded by caller
        start, end = self._quiet_hours
        hour = now.hour
        if start <= end:
            # Same-day window, e.g. 08:00 – 18:00.
            return start <= hour < end
        # Overnight window, e.g. 22:00 – 06:00.
        return hour >= start or hour < end
