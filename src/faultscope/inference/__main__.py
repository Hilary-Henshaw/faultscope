"""CLI entry point for the FaultScope inference service.

Starts the FastAPI application under uvicorn using settings from
environment variables / ``.env``.

Invocation
----------
    python -m faultscope.inference
    python -m faultscope.inference --workers 2 --port 8080
    faultscope-serve  # via pyproject.toml script entry point

Options
-------
--host <str>
    Bind host.  Overrides ``FAULTSCOPE_INFERENCE_HOST``.
--port <int>
    Bind port.  Overrides ``FAULTSCOPE_INFERENCE_PORT``.
--workers <int>
    Number of uvicorn workers.  Overrides
    ``FAULTSCOPE_INFERENCE_WORKERS``.
--log-level <str>
    Uvicorn log level.  Defaults to ``info``.

Exit codes
----------
0   Server exited normally (e.g. SIGTERM).
1   Configuration error or unhandled exception at startup.
"""

from __future__ import annotations

import argparse
import sys

from faultscope.common.exceptions import ConfigurationError
from faultscope.common.logging import configure_logging, get_logger


def _parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="faultscope-serve",
        description="Start the FaultScope inference service.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override bind host (default from config).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override bind port (default from config).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of uvicorn workers (default from config).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Uvicorn log level.  Defaults to info.",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Parse CLI arguments, build the app, and start uvicorn."""
    args = _parse_args()

    # Configure structlog before any imports that might log.
    configure_logging(
        level=args.log_level.upper(),
        fmt="json",
    )
    log = get_logger(__name__)

    try:
        from faultscope.inference.api.app import create_app
        from faultscope.inference.config import InferenceConfig

        config = InferenceConfig()

        if args.host is not None:
            object.__setattr__(config, "host", args.host)
        if args.port is not None:
            object.__setattr__(config, "port", args.port)
        if args.workers is not None:
            object.__setattr__(config, "workers", args.workers)

    except ConfigurationError as exc:
        log.error(
            "inference_configuration_error",
            error=str(exc),
            context=exc.context,
        )
        sys.exit(1)
    except Exception as exc:
        log.error(
            "inference_startup_error",
            error=str(exc),
            exc_info=True,
        )
        sys.exit(1)

    import uvicorn

    log.info(
        "inference_server_starting",
        host=config.host,
        port=config.port,
        workers=config.workers,
    )

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level=args.log_level,
        access_log=False,
        # Disable uvicorn's default log config so that structlog
        # remains the sole log handler.
        log_config=None,
    )


if __name__ == "__main__":
    main()
