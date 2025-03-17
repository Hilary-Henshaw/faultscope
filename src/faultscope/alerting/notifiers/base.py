"""Abstract base for all FaultScope notification channels.

Concrete notifiers (email, Slack, webhook) extend ``BaseNotifier`` and
implement ``send``.  The contract requires that ``send`` never raises —
any delivery failure must be logged internally so that a bad notifier
cannot crash the coordinator.

Usage::

    class MyNotifier(BaseNotifier):
        async def send(self, payload: NotificationPayload) -> None:
            # deliver notification; log errors, never raise
            ...

        @property
        def channel_name(self) -> str:
            return "my-channel"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime

from faultscope.alerting.engine.evaluator import TriggeredIncident
from faultscope.alerting.rules import Severity


@dataclass
class NotificationPayload:
    """Aggregated notification payload sent to every active channel.

    Attributes
    ----------
    machine_id:
        The machine for which incidents were triggered.
    severity:
        Highest severity among all incidents in this payload.
    title:
        Short summary title for the notification.
    incidents:
        All deduplicated incidents grouped for this machine.
    triggered_at:
        UTC timestamp of the first incident in the group.
    """

    machine_id: str
    severity: Severity
    title: str
    incidents: list[TriggeredIncident]
    triggered_at: datetime = field(
        default_factory=lambda: datetime.now(tz=UTC)
    )


class BaseNotifier(ABC):
    """Abstract base class for all notification channels.

    All implementations must satisfy the following contract:

    - ``send()`` is ``async`` and must never propagate an exception;
      failures must be logged internally.
    - ``channel_name`` returns a stable lowercase identifier used in
      log entries and metrics.
    """

    @abstractmethod
    async def send(self, payload: NotificationPayload) -> None:
        """Deliver the notification payload to the channel.

        Implementations must catch all delivery exceptions internally
        and log them.  This method must not raise under any
        circumstances to avoid disrupting the coordinator dispatch loop.

        Parameters
        ----------
        payload:
            The aggregated notification payload to deliver.
        """
        ...

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Return a stable lowercase identifier for this channel.

        Returns
        -------
        str
            E.g. ``"email"``, ``"slack"``, ``"webhook"``.
        """
        ...
