"""Entry point for the FaultScope streaming service.

Usage::

    # Via the installed script entry-point:
    faultscope-stream

    # Or directly:
    python -m faultscope.streaming

The service:

1. Reads configuration from environment variables (``FAULTSCOPE_*``)
   and an optional ``.env`` file.
2. Configures structured logging.
3. Starts a Prometheus metrics HTTP server on ``config.metrics_port``.
4. Creates and runs ``FeaturePipeline`` until ``SIGTERM`` or
   ``SIGINT`` is received, at which point it shuts down gracefully.
"""

from __future__ import annotations

import asyncio
import signal
import sys

import structlog
from prometheus_client import start_http_server

from faultscope.common.logging import configure_logging, get_logger
from faultscope.streaming.config import StreamingConfig
from faultscope.streaming.pipeline import FeaturePipeline

log: structlog.stdlib.BoundLogger = get_logger(__name__)


async def main_async() -> None:
    """Asynchronous entry point.

    Sets up configuration, logging, Prometheus, and the pipeline.
    Installs ``SIGTERM`` and ``SIGINT`` handlers that trigger a clean
    shutdown of the pipeline before the process exits.
    """
    config = StreamingConfig()

    configure_logging(config.log_level, config.log_format)

    log.info(
        "streaming.service_starting",
        metrics_port=config.metrics_port,
        kafka_servers=config.kafka_bootstrap_servers,
        consumer_group=config.kafka_consumer_group,
        input_topic=config.topic_sensor_readings,
        output_topic=config.topic_computed_features,
        windows=config.rolling_windows_s,
    )

    # ── Prometheus metrics server ─────────────────────────────────────
    try:
        start_http_server(config.metrics_port)
        log.info(
            "streaming.metrics_server_started",
            port=config.metrics_port,
        )
    except OSError as exc:
        log.error(
            "streaming.metrics_server_failed",
            port=config.metrics_port,
            error=str(exc),
        )
        sys.exit(1)

    pipeline = FeaturePipeline(config)

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _request_shutdown(sig: signal.Signals) -> None:
        log.info(
            "streaming.shutdown_requested",
            signal=sig.name,
        )
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(
            sig,
            lambda s=sig: _request_shutdown(s),  # type: ignore[misc]
        )

    # ── Start pipeline ────────────────────────────────────────────────
    try:
        await pipeline.start()
    except Exception as exc:  # noqa: BLE001
        log.error(
            "streaming.start_failed",
            error=str(exc),
            exc_info=True,
        )
        sys.exit(1)

    # ── Run until shutdown is requested ──────────────────────────────
    run_task = asyncio.create_task(pipeline.run(), name="pipeline_run")
    shutdown_task = asyncio.create_task(
        shutdown_event.wait(), name="shutdown_watcher"
    )

    done, pending = await asyncio.wait(
        {run_task, shutdown_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel any remaining tasks.
    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
            pass

    # Surface pipeline errors.
    for task in done:
        pipeline_exc = task.exception()
        if pipeline_exc is not None and task is run_task:
            log.error(
                "streaming.pipeline_error",
                error=str(pipeline_exc),
                exc_info=pipeline_exc,
            )

    # ── Graceful shutdown ─────────────────────────────────────────────
    log.info("streaming.shutting_down")
    try:
        await pipeline.stop()
    except Exception as exc:  # noqa: BLE001
        log.error(
            "streaming.stop_error",
            error=str(exc),
            exc_info=True,
        )

    log.info("streaming.service_stopped")


def main() -> None:
    """Synchronous entry point called by the installed script.

    Delegates to ``asyncio.run`` so the event loop is created and
    torn down cleanly.
    """
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
