"""HTML email notifier for the FaultScope alerting service.

Sends formatted HTML alert emails via SMTP using ``aiosmtplib``.
STARTTLS is used when ``smtp_port`` is 587; SSL is used for port 465.
For any other port, a plain connection is attempted.

No credentials are hardcoded.  All authentication material is passed
via constructor arguments that the caller must source from
``AlertingConfig`` secrets.
"""

from __future__ import annotations

import html
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import aiosmtplib

from faultscope.alerting.engine.evaluator import TriggeredIncident
from faultscope.alerting.notifiers.base import (
    BaseNotifier,
    NotificationPayload,
)
from faultscope.alerting.rules import Severity
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

# ANSI-style hex colours used in the HTML body.
_SEVERITY_COLOURS: dict[Severity, str] = {
    Severity.CRITICAL: "#c0392b",
    Severity.WARNING: "#e67e22",
    Severity.INFO: "#2980b9",
}
_SEVERITY_LABELS: dict[Severity, str] = {
    Severity.CRITICAL: "CRITICAL",
    Severity.WARNING: "WARNING",
    Severity.INFO: "INFO",
}


class EmailNotifier(BaseNotifier):
    """Send HTML email alerts via SMTP using ``aiosmtplib``.

    Parameters
    ----------
    smtp_host:
        Hostname of the SMTP server.
    smtp_port:
        Port number.  587 → STARTTLS, 465 → SSL, other → plain.
    username:
        SMTP authentication username.
    password:
        SMTP authentication password (sourced from a ``SecretStr``
        via the caller — never hardcoded).
    from_addr:
        Sender email address shown in the ``From:`` header.
    recipients:
        List of recipient email addresses.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        recipients: list[str],
    ) -> None:
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._username = username
        self._password = password
        self._from_addr = from_addr
        self._recipients = list(recipients)

    @property
    def channel_name(self) -> str:
        """Return the channel identifier."""
        return "email"

    async def send(self, payload: NotificationPayload) -> None:
        """Build and dispatch an HTML email for ``payload``.

        Catches all SMTP and network exceptions internally and logs them
        at ``ERROR`` level.  Never raises.

        Parameters
        ----------
        payload:
            Aggregated notification payload for one machine.
        """
        if not self._recipients:
            _log.warning(
                "email_notifier_no_recipients",
                machine_id=payload.machine_id,
            )
            return

        subject = (
            f"[FaultScope {_SEVERITY_LABELS[payload.severity]}] "
            f"{payload.machine_id}: {payload.title}"
        )
        html_body = self._build_html_body(payload)

        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self._from_addr
        message["To"] = ", ".join(self._recipients)
        message.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            use_tls = self._smtp_port == 465
            start_tls = self._smtp_port == 587
            await aiosmtplib.send(
                message,
                hostname=self._smtp_host,
                port=self._smtp_port,
                username=self._username or None,
                password=self._password or None,
                use_tls=use_tls,
                start_tls=start_tls,
            )
            _log.info(
                "email_sent",
                machine_id=payload.machine_id,
                severity=payload.severity.value,
                recipients=self._recipients,
            )
        except aiosmtplib.SMTPException as exc:
            _log.error(
                "email_send_failed_smtp",
                machine_id=payload.machine_id,
                error=str(exc),
                smtp_host=self._smtp_host,
                smtp_port=self._smtp_port,
            )
        except OSError as exc:
            _log.error(
                "email_send_failed_network",
                machine_id=payload.machine_id,
                error=str(exc),
                smtp_host=self._smtp_host,
            )

    def _build_html_body(self, payload: NotificationPayload) -> str:
        """Render a complete HTML email body for ``payload``.

        Parameters
        ----------
        payload:
            Notification payload containing machine info and incidents.

        Returns
        -------
        str
            Well-formed HTML string suitable for the ``text/html`` MIME
            part of an email.
        """
        colour = _SEVERITY_COLOURS[payload.severity]
        severity_label = _SEVERITY_LABELS[payload.severity]
        triggered_str = payload.triggered_at.strftime("%Y-%m-%d %H:%M:%S UTC")

        incident_rows = "".join(
            self._render_incident_row(inc) for inc in payload.incidents
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FaultScope Alert</title>
</head>
<body style="font-family: Arial, sans-serif; color: #333; margin: 0;
             padding: 0; background-color: #f5f5f5;">
  <table width="100%" cellpadding="0" cellspacing="0"
         style="background-color: #f5f5f5; padding: 20px;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0"
               style="background-color: #ffffff; border-radius: 6px;
                      box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
          <!-- Header -->
          <tr>
            <td style="background-color: {colour}; padding: 20px 30px;
                       border-radius: 6px 6px 0 0;">
              <h1 style="color: #ffffff; margin: 0; font-size: 20px;">
                &#9888; FaultScope Alert &mdash; {severity_label}
              </h1>
            </td>
          </tr>
          <!-- Summary -->
          <tr>
            <td style="padding: 24px 30px 0 30px;">
              <table width="100%" cellpadding="6" cellspacing="0"
                     style="border-collapse: collapse;
                            background-color: #fafafa;
                            border: 1px solid #e0e0e0;
                            border-radius: 4px;">
                <tr>
                  <td style="font-weight: bold; width: 140px;">
                    Machine
                  </td>
                  <td>{html.escape(payload.machine_id)}</td>
                </tr>
                <tr style="background-color: #f0f0f0;">
                  <td style="font-weight: bold;">Severity</td>
                  <td style="color: {colour}; font-weight: bold;">
                    {severity_label}
                  </td>
                </tr>
                <tr>
                  <td style="font-weight: bold;">Triggered At</td>
                  <td>{triggered_str}</td>
                </tr>
                <tr style="background-color: #f0f0f0;">
                  <td style="font-weight: bold;">Incidents</td>
                  <td>{len(payload.incidents)}</td>
                </tr>
              </table>
            </td>
          </tr>
          <!-- Incident details -->
          <tr>
            <td style="padding: 20px 30px;">
              <h2 style="font-size: 16px; color: #555;
                         border-bottom: 2px solid {colour};
                         padding-bottom: 6px;">
                Triggered Rules
              </h2>
              <table width="100%" cellpadding="8" cellspacing="0"
                     style="border-collapse: collapse;
                            border: 1px solid #e0e0e0;">
                <thead>
                  <tr style="background-color: {colour}; color: #fff;">
                    <th align="left">Rule</th>
                    <th align="left">Severity</th>
                    <th align="left">RUL (cycles)</th>
                    <th align="left">Anomaly Score</th>
                    <th align="left">Health</th>
                  </tr>
                </thead>
                <tbody>
                  {incident_rows}
                </tbody>
              </table>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="padding: 12px 30px 20px 30px;
                       border-top: 1px solid #e0e0e0;
                       font-size: 12px; color: #999;">
              This alert was generated automatically by FaultScope
              predictive maintenance. Do not reply to this message.
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

    @staticmethod
    def _render_incident_row(inc: TriggeredIncident) -> str:
        """Render a single ``<tr>`` for an incident in the email table.

        Parameters
        ----------
        inc:
            The incident to render.

        Returns
        -------
        str
            An HTML ``<tr>`` string.
        """
        row_colour = _SEVERITY_COLOURS[inc.severity]
        rul = inc.details.get("rul_cycles", "—")
        anomaly = inc.details.get("anomaly_score", "—")
        health = inc.details.get("health_label", "—")
        rul_str = f"{rul:.1f}" if isinstance(rul, float) else str(rul)
        anomaly_str = (
            f"{anomaly:.3f}" if isinstance(anomaly, float) else str(anomaly)
        )
        return (
            f"<tr>"
            f"<td>{html.escape(inc.rule.rule_name)}</td>"
            f"<td style='color: {row_colour}; font-weight: bold;'>"
            f"{html.escape(inc.severity.value.upper())}</td>"
            f"<td>{html.escape(rul_str)}</td>"
            f"<td>{html.escape(anomaly_str)}</td>"
            f"<td>{html.escape(str(health))}</td>"
            f"</tr>"
        )
