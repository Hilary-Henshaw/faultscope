"""Structured logging configuration for FaultScope services.

Call ``configure_logging`` exactly once during service startup before
any other code creates loggers.  Afterwards, obtain per-module loggers
via ``get_logger(__name__)``.

Example::

    from faultscope.common.logging import configure_logging, get_logger

    configure_logging(level="INFO", fmt="json")
    log = get_logger(__name__)
    log.info("service_started", port=8000)
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str, fmt: str) -> None:
    """Configure structlog for the service.

    Must be called once at startup before any loggers are obtained.
    Configures the standard-library ``logging`` module as the backend
    so that third-party libraries (SQLAlchemy, aiokafka, etc.) are also
    captured with the same renderer.

    Parameters
    ----------
    level:
        Log level string accepted by ``logging``, e.g. ``"INFO"``,
        ``"DEBUG"``, ``"WARNING"``, ``"ERROR"``.
    fmt:
        Renderer selection.  ``"json"`` emits newline-delimited JSON
        (suitable for log aggregators in production).  Any other value
        falls back to the human-friendly ``ConsoleRenderer`` used
        during local development.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors applied to every log record regardless of
    # whether it originated from structlog or stdlib logging.
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
    ]

    if fmt == "json":
        renderer: structlog.types.Processor = (
            structlog.processors.JSONRenderer()
        )
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            # Prepare event dict for the stdlib formatter so that the
            # ProcessorFormatter can re-render it.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # These processors run only on stdlib-originated records.
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    # Remove any handlers added by the runtime before we configure.
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Suppress noisy third-party debug output unless the service
    # explicitly requests DEBUG level.
    for noisy in ("aiokafka", "sqlalchemy.engine", "asyncio"):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if log_level == logging.DEBUG else logging.WARNING
        )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module name.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    structlog.stdlib.BoundLogger
        A logger that supports keyword-argument-based structured events,
        e.g. ``log.info("event", machine_id="M001")``.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]
