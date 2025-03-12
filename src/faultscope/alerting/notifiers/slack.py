"""Slack Block Kit notifier for the FaultScope alerting service.

Posts richly formatted messages to a Slack channel via an incoming
webhook URL.  Block Kit is used so that fields, severity badges, and
incident detail tables render in a visually distinct layout within Slack.

No credentials are hardcoded.  The webhook URL is passed at construction
time by the caller and sourced from ``AlertingConfig.slack_webhook_url``.
"""

from __future__ import annotations

import json

import httpx

from faultscope.alerting.notifiers.base import (
    BaseNotifier,
    NotificationPayload,
)
from faultscope.alerting.rules import Severity
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

_SEVERITY_EMOJI: dict[Severity, str] = {
    Severity.CRITICAL: ":rotating_light:",
    Severity.WARNING: ":warning:",
    Severity.INFO: ":information_source:",
}
_SEVERITY_COLOUR: dict[Severity, str] = {
    Severity.CRITICAL: "#c0392b",
    Severity.WARNING: "#e67e22",
    Severity.INFO: "#2980b9",
}

_HTTP_TIMEOUT_S: float = 10.0


class SlackNotifier(BaseNotifier):
    """Post Block Kit alert messages to Slack via incoming webhook.

    Parameters
    ----------
    webhook_url:
        Slack incoming webhook URL.  Sourced from
        ``AlertingConfig.slack_webhook_url.get_secret_value()``.
    channel:
        Slack channel override, e.g. ``"#equipment-alerts"``.
        Many webhook configurations pin the channel server-side;
        this value is included in the payload for clarity but may
        be ignored by Slack depending on the webhook setup.
    mention_handle:
        User group or user handle to mention in critical alerts,
        e.g. ``"@on-call"``.  Empty string disables mentions.
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str,
        mention_handle: str = "",
    ) -> None:
        self._webhook_url = webhook_url
        self._channel = channel
        self._mention_handle = mention_handle

    @property
    def channel_name(self) -> str:
        """Return the channel identifier."""
        return "slack"

    async def send(self, payload: NotificationPayload) -> None:
        """Build and post a Block Kit message for ``payload``.

        Catches all HTTP and network exceptions internally.
        Never raises.

        Parameters
        ----------
        payload:
            Aggregated notification payload for one machine.
        """
        if not self._webhook_url:
            _log.warning(
                "slack_notifier_no_webhook_url",
                machine_id=payload.machine_id,
            )
            return

        blocks = self._build_blocks(payload)
        body = {
            "channel": self._channel,
            "blocks": blocks,
            # Fallback plain-text for notifications / screen readers.
            "text": (
                f"{_SEVERITY_EMOJI[payload.severity]} "
                f"FaultScope {payload.severity.value.upper()}: "
                f"{payload.machine_id} — {payload.title}"
            ),
        }

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
                response = await client.post(
                    self._webhook_url,
                    content=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

            _log.info(
                "slack_message_sent",
                machine_id=payload.machine_id,
                severity=payload.severity.value,
                channel=self._channel,
            )
        except httpx.HTTPStatusError as exc:
            _log.error(
                "slack_send_failed_http_status",
                machine_id=payload.machine_id,
                status_code=exc.response.status_code,
                error=str(exc),
            )
        except httpx.RequestError as exc:
            _log.error(
                "slack_send_failed_request",
                machine_id=payload.machine_id,
                error=str(exc),
            )

    def _build_blocks(
        self,
        payload: NotificationPayload,
    ) -> list[dict[str, object]]:
        """Construct Slack Block Kit JSON for ``payload``.

        Produces:
        - A header block with severity emoji and title.
        - A section block with machine ID, severity, triggered time,
          and optional mention.
        - One section block per triggered incident listing rule name
          and key metric values.
        - A divider footer.

        Parameters
        ----------
        payload:
            Notification payload to render.

        Returns
        -------
        list[dict[str, object]]
            Block Kit block list ready for the Slack API payload.
        """
        emoji = _SEVERITY_EMOJI[payload.severity]
        sev_label = payload.severity.value.upper()
        triggered_str = payload.triggered_at.strftime("%Y-%m-%d %H:%M:%S UTC")

        mention_text = (
            f" {self._mention_handle}" if self._mention_handle else ""
        )

        blocks: list[dict[str, object]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": (
                        f"{emoji} FaultScope {sev_label} Alert{mention_text}"
                    ),
                    "emoji": True,
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Machine ID:*\n`{payload.machine_id}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": (f"*Severity:*\n{emoji} `{sev_label}`"),
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Triggered At:*\n{triggered_str}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": (
                            f"*Incidents:*\n"
                            f"{len(payload.incidents)} rule(s) fired"
                        ),
                    },
                ],
            },
            {"type": "divider"},
        ]

        for inc in payload.incidents:
            inc_emoji = _SEVERITY_EMOJI[inc.severity]
            rul = inc.details.get("rul_cycles", "—")
            anomaly = inc.details.get("anomaly_score", "—")
            health = inc.details.get("health_label", "—")
            rul_str = f"{rul:.1f}" if isinstance(rul, float) else str(rul)
            anomaly_str = (
                f"{anomaly:.3f}"
                if isinstance(anomaly, float)
                else str(anomaly)
            )
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"{inc_emoji} *{inc.rule.rule_name}*"
                            f" — `{inc.severity.value.upper()}`\n"
                            f"_{inc.rule.description}_"
                        ),
                    },
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*RUL (cycles):*\n{rul_str}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": (f"*Anomaly Score:*\n{anomaly_str}"),
                        },
                        {
                            "type": "mrkdwn",
                            "text": (f"*Health Label:*\n`{health}`"),
                        },
                        {
                            "type": "mrkdwn",
                            "text": (f"*Rule ID:*\n`{inc.rule.rule_id}`"),
                        },
                    ],
                }
            )

        blocks.append({"type": "divider"})
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": (
                            "Generated by *FaultScope* predictive "
                            "maintenance platform."
                        ),
                    }
                ],
            }
        )

        return blocks
