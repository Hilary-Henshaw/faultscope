"""FastAPI application factory for the FaultScope alerting service.

Call ``create_app`` with a resolved ``AlertingConfig`` to obtain the
fully configured ``FastAPI`` application instance.  The factory:

- Creates an ``asyncpg`` connection pool on startup.
- Wires up all configured notifiers.
- Instantiates the ``IncidentCoordinator`` and stores it on
  ``app.state`` for injection into route handlers.
- Registers all API routers.
- Adds a ``/health`` liveness probe.

Usage::

    from faultscope.alerting.config import AlertingConfig
    from faultscope.alerting.api.app import create_app

    app = create_app(AlertingConfig())
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from faultscope.alerting.api.routes.incidents import (
    machines_router,
)
from faultscope.alerting.api.routes.incidents import (
    router as incidents_router,
)
from faultscope.alerting.api.routes.rules import router as rules_router
from faultscope.alerting.config import AlertingConfig
from faultscope.alerting.coordinator import IncidentCoordinator
from faultscope.alerting.notifiers.base import BaseNotifier
from faultscope.alerting.notifiers.email import EmailNotifier
from faultscope.alerting.notifiers.slack import SlackNotifier
from faultscope.alerting.notifiers.webhook import WebhookNotifier
from faultscope.common.logging import configure_logging, get_logger

_log = get_logger(__name__)


def _build_notifiers(config: AlertingConfig) -> list[BaseNotifier]:
    """Instantiate all enabled notification channels from config.

    A channel is considered enabled when its primary configuration
    field is non-empty (e.g. ``email_recipients`` for email,
    ``slack_webhook_url`` for Slack, ``webhook_url`` for HTTP webhook).

    Parameters
    ----------
    config:
        Resolved ``AlertingConfig`` instance.

    Returns
    -------
    list[BaseNotifier]
        All enabled notifier instances.
    """
    notifiers: list[BaseNotifier] = []

    if config.email_recipients and config.email_smtp_host:
        notifiers.append(
            EmailNotifier(
                smtp_host=config.email_smtp_host,
                smtp_port=config.email_smtp_port,
                username=config.email_username,
                password=config.email_password.get_secret_value(),
                from_addr=config.email_from,
                recipients=config.email_recipients,
            )
        )
        _log.info("email_notifier_enabled")

    slack_url = config.slack_webhook_url.get_secret_value()
    if slack_url:
        notifiers.append(
            SlackNotifier(
                webhook_url=slack_url,
                channel=config.slack_channel,
                mention_handle=config.slack_mention,
            )
        )
        _log.info("slack_notifier_enabled")

    if config.webhook_url:
        notifiers.append(WebhookNotifier(webhook_url=config.webhook_url))
        _log.info("webhook_notifier_enabled")

    if not notifiers:
        _log.warning(
            "no_notifiers_configured",
            hint=(
                "Set FAULTSCOPE_EMAIL_RECIPIENTS, "
                "FAULTSCOPE_SLACK_WEBHOOK_URL, or "
                "FAULTSCOPE_WEBHOOK_URL to enable notifications."
            ),
        )

    return notifiers


def create_app(config: AlertingConfig) -> FastAPI:
    """Build and configure the alerting service FastAPI application.

    Parameters
    ----------
    config:
        Resolved ``AlertingConfig`` containing all service settings.

    Returns
    -------
    FastAPI
        Fully configured application ready to serve requests.
    """
    configure_logging(level=config.log_level, fmt=config.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Manage service lifecycle: startup and graceful shutdown."""
        _log.info(
            "alerting_service_starting",
            host=config.host,
            port=config.port,
        )

        pool: asyncpg.Pool = await asyncpg.create_pool(  # type: ignore[type-arg]
            dsn=config.db_async_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        _log.info("db_pool_created")

        notifiers = _build_notifiers(config)
        coordinator = IncidentCoordinator(
            config=config,
            db_pool=pool,
            notifiers=notifiers,
        )

        app.state.coordinator = coordinator
        app.state.db_pool = pool
        app.state.config = config

        _log.info(
            "alerting_service_started",
            notifiers=[n.channel_name for n in notifiers],
        )

        yield

        _log.info("alerting_service_stopping")
        await pool.close()
        _log.info("alerting_service_stopped")

    application = FastAPI(
        title="FaultScope Alerting Service",
        description=(
            "Real-time predictive maintenance alerting: "
            "rule evaluation, incident lifecycle, and multi-channel "
            "notification dispatch."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS — restrict origins in production via environment variables.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
        allow_credentials=False,
    )

    # Routers
    application.include_router(incidents_router)
    application.include_router(machines_router)
    application.include_router(rules_router)

    @application.get(
        "/health",
        tags=["Health"],
        summary="Liveness probe",
    )
    async def health() -> dict[str, str]:
        """Return service liveness status.

        Returns
        -------
        dict[str, str]
            ``{"status": "ok", "service": "faultscope-alerting"}``.
        """
        return {
            "status": "ok",
            "service": "faultscope-alerting",
        }

    @application.get(
        "/ready",
        tags=["Health"],
        summary="Readiness probe",
    )
    async def ready() -> dict[str, object]:
        """Return service readiness status including DB connectivity.

        Returns
        -------
        dict[str, object]
            Readiness details including database pool state.
        """
        pool: asyncpg.Pool = application.state.db_pool  # type: ignore[type-arg]
        pool_size = pool.get_size()
        return {
            "status": "ready",
            "service": "faultscope-alerting",
            "db_pool_size": pool_size,
        }

    return application
