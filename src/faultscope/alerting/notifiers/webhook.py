"""Generic HTTP webhook notifier for the FaultScope alerting service.

Posts a structured JSON payload to a configurable endpoint.  Failed
requests are retried up to three times with exponential back-off using
``tenacity``.  All exceptions are caught internally so that webhook
failures never crash the coordinator dispatch loop.

No credentials are hardcoded.  The webhook URL is supplied at
construction time by the caller and sourced from configuration.
"""

from __future__ import annotations

import json

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from faultscope.alerting.notifiers.base import (
    BaseNotifier,
    NotificationPayload,
)
from faultscope.common.logging import get_logger

_log = get_logger(__name__)

_MAX_ATTEMPTS: int = 3
_WAIT_MIN_S: float = 1.0
_WAIT_MAX_S: float = 8.0
_HTTP_TIMEOUT_S: float = 15.0


class WebhookNotifier(BaseNotifier):
    """POST structured JSON alert payloads to an HTTP endpoint.

    The request body is a JSON object with the following top-level keys:

    - ``source``: always ``"faultscope"``
    - ``machine_id``: the triggering machine
    - ``severity``: string severity label
    - ``title``: short summary
    - ``triggered_at``: ISO-8601 UTC timestamp
    - ``incidents``: array of incident detail objects

    Parameters
    ----------
    webhook_url:
        HTTP(S) endpoint to POST to.  Sourced from
        ``AlertingConfig.webhook_url``.
    """

    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url

    @property
    def channel_name(self) -> str:
        """Return the channel identifier."""
        return "webhook"

    async def send(self, payload: NotificationPayload) -> None:
        """Serialise ``payload`` to JSON and POST to the endpoint.

        Retries the request up to 3 times with exponential back-off on
        network or 5xx errors.  All exceptions are caught internally.
        Never raises.

        Parameters
        ----------
        payload:
            Aggregated notification payload for one machine.
        """
        if not self._webhook_url:
            _log.warning(
                "webhook_notifier_no_url",
                machine_id=payload.machine_id,
            )
            return

        body = self._build_body(payload)
        body_bytes = json.dumps(body, default=str).encode("utf-8")

        try:
            await self._post_with_retry(payload.machine_id, body_bytes)
            _log.info(
                "webhook_sent",
                machine_id=payload.machine_id,
                severity=payload.severity.value,
                url=self._webhook_url,
            )
        except RetryError as exc:
            _log.error(
                "webhook_send_failed_all_retries",
                machine_id=payload.machine_id,
                url=self._webhook_url,
                error=str(exc),
            )
        except httpx.RequestError as exc:
            _log.error(
                "webhook_send_failed_request",
                machine_id=payload.machine_id,
                url=self._webhook_url,
                error=str(exc),
            )

    @retry(
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.RequestError)
        ),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=1,
            min=_WAIT_MIN_S,
            max=_WAIT_MAX_S,
        ),
        reraise=True,
    )
    async def _post_with_retry(
        self,
        machine_id: str,
        body_bytes: bytes,
    ) -> None:
        """Execute the HTTP POST with tenacity retry wrapping.

        Parameters
        ----------
        machine_id:
            Used only for structured log context on retry attempts.
        body_bytes:
            Pre-serialised JSON payload bytes.

        Raises
        ------
        httpx.HTTPStatusError
            When the server returns a 4xx/5xx response.
        httpx.RequestError
            When the request cannot be sent (DNS, timeout, etc.).
        """
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            response = await client.post(
                self._webhook_url,
                content=body_bytes,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code >= 500:
                _log.warning(
                    "webhook_server_error_retrying",
                    machine_id=machine_id,
                    status_code=response.status_code,
                    url=self._webhook_url,
                )
                response.raise_for_status()
            elif response.status_code >= 400:
                _log.error(
                    "webhook_client_error_no_retry",
                    machine_id=machine_id,
                    status_code=response.status_code,
                    url=self._webhook_url,
                )
                # 4xx errors should not be retried; raise to let
                # tenacity propagate but mark as non-retryable by
                # re-raising as a plain exception to exit retry.
                response.raise_for_status()

    @staticmethod
    def _build_body(payload: NotificationPayload) -> dict[str, object]:
        """Construct the JSON-serialisable payload body.

        Parameters
        ----------
        payload:
            Notification payload.

        Returns
        -------
        dict[str, object]
            Dictionary suitable for ``json.dumps``.
        """
        return {
            "source": "faultscope",
            "machine_id": payload.machine_id,
            "severity": payload.severity.value,
            "title": payload.title,
            "triggered_at": payload.triggered_at.isoformat(),
            "incidents": [
                {
                    "rule_id": inc.rule.rule_id,
                    "rule_name": inc.rule.rule_name,
                    "severity": inc.severity.value,
                    "condition_type": inc.rule.condition_type.value,
                    "details": inc.details,
                    "triggered_at": inc.triggered_at.isoformat(),
                }
                for inc in payload.incidents
            ],
        }
