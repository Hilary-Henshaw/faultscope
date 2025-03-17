"""Entry point for the FaultScope alerting service.

Starts two concurrent tasks:
1. The FastAPI HTTP server (uvicorn).
2. The Kafka ``PredictionEventConsumer`` consuming prediction events.

Both tasks run until a ``SIGINT`` or ``SIGTERM`` signal is received,
at which point the consumer is asked to stop and the server shuts down
gracefully.

Usage::

    python -m faultscope.alerting
    # or via the project script:
    faultscope-alerts
"""

from __future__ import annotations

import asyncio
import signal

import asyncpg
import uvicorn

from faultscope.alerting.api.app import _build_notifiers, create_app
from faultscope.alerting.config import AlertingConfig
from faultscope.alerting.consumer import PredictionEventConsumer
from faultscope.alerting.coordinator import IncidentCoordinator
from faultscope.common.logging import configure_logging, get_logger

_log = get_logger(__name__)


async def _run(config: AlertingConfig) -> None:
    """Bootstrap all service components and run until shutdown.

    Parameters
    ----------
    config:
        Resolved ``AlertingConfig`` instance.
    """
    configure_logging(level=config.log_level, fmt=config.log_format)

    _log.info(
        "alerting_service_bootstrap",
        host=config.host,
        port=config.port,
        kafka=config.kafka_bootstrap_servers,
    )

    pool: asyncpg.Pool = await asyncpg.create_pool(  # type: ignore[type-arg]
        dsn=config.db_async_url,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    _log.info("db_pool_ready")

    notifiers = _build_notifiers(config)
    coordinator = IncidentCoordinator(
        config=config,
        db_pool=pool,
        notifiers=notifiers,
    )

    consumer = PredictionEventConsumer(
        config=config,
        coordinator=coordinator,
    )

    app = create_app(config)
    # Inject already-created coordinator so the lifespan does not
    # create a second pool.
    app.state.coordinator = coordinator
    app.state.db_pool = pool
    app.state.config = config

    uv_config = uvicorn.Config(
        app=app,
        host=config.host,
        port=config.port,
        log_config=None,  # structlog handles all logging
        access_log=False,
    )
    server = uvicorn.Server(config=uv_config)

    # ------------------------------------------------------------------ #
    # Graceful shutdown on SIGINT / SIGTERM
    # ------------------------------------------------------------------ #
    shutdown_event = asyncio.Event()

    def _handle_signal(sig: int) -> None:
        _log.info(
            "shutdown_signal_received",
            signal=signal.Signals(sig).name,
        )
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, sig)

    async def _wait_for_shutdown() -> None:
        await shutdown_event.wait()
        await consumer.stop()
        server.should_exit = True

    try:
        await asyncio.gather(
            consumer.run(),
            server.serve(),
            _wait_for_shutdown(),
            return_exceptions=True,
        )
    finally:
        await pool.close()
        _log.info("alerting_service_shutdown_complete")


def main() -> None:
    """Synchronous entry point invoked by the ``faultscope-alerts`` script.

    Loads configuration from the environment and starts the event loop.
    """
    config = AlertingConfig()
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
