"""CLI entry point for the FaultScope retraining pipeline.

Invocation
----------
    python -m faultscope.retraining [--force] [--reason=manual]

Options
-------
--force
    Skip drift detection and retrain unconditionally.
--reason=<str>
    Human-readable trigger reason.  Defaults to ``"manual"``.
--log-level=<str>
    Logging level: DEBUG, INFO, WARNING, ERROR.  Defaults to ``INFO``.
--log-format=<str>
    Log renderer: ``json`` (production) or ``console`` (dev).
    Defaults to ``json``.

Exit codes
----------
0   Pipeline completed successfully (retrain may or may not have run).
1   Pipeline failed with an unhandled exception.
2   Configuration error (missing required env variables).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from faultscope.common.exceptions import ConfigurationError
from faultscope.common.logging import configure_logging, get_logger


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="faultscope-retrain",
        description="Run the FaultScope MLOps retraining pipeline.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=("Skip drift detection and retrain unconditionally."),
    )
    parser.add_argument(
        "--reason",
        default="manual",
        help=(
            "Human-readable trigger reason stored in the drift_event "
            "log.  Defaults to 'manual'."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Structlog/stdlib log level.  Defaults to INFO.",
    )
    parser.add_argument(
        "--log-format",
        default="json",
        choices=["json", "console"],
        help=(
            "Log renderer: 'json' for production, 'console' for dev."
            "  Defaults to json."
        ),
    )
    return parser.parse_args(argv)


async def _run(force: bool, reason: str) -> dict[str, object]:
    """Async implementation of the retraining run."""
    # Import here so that configuration errors surface after logging
    # is configured.
    from faultscope.retraining.config import RetrainingConfig
    from faultscope.retraining.pipeline import RetrainingOrchestrator

    config = RetrainingConfig()
    orchestrator = RetrainingOrchestrator(config)
    return await orchestrator.run(reason=reason, force=force)


def main() -> None:
    """Parse CLI arguments and execute the retraining pipeline."""
    args = _parse_args()
    configure_logging(level=args.log_level, fmt=args.log_format)
    log = get_logger(__name__)

    log.info(
        "retraining_cli_start",
        force=args.force,
        reason=args.reason,
    )

    try:
        summary = asyncio.run(_run(force=args.force, reason=args.reason))
    except ConfigurationError as exc:
        log.error(
            "retraining_configuration_error",
            error=str(exc),
            context=exc.context,
        )
        sys.exit(2)
    except KeyboardInterrupt:
        log.info("retraining_interrupted")
        sys.exit(0)
    except Exception as exc:
        log.error(
            "retraining_unhandled_error",
            error=str(exc),
            exc_info=True,
        )
        sys.exit(1)

    print(json.dumps(summary, indent=2, default=str))  # noqa: T201
    log.info(
        "retraining_cli_complete",
        triggered=summary.get("triggered"),
        rul_promoted=summary.get("rul_promoted"),
        health_promoted=summary.get("health_promoted"),
        duration_s=summary.get("duration_s"),
    )


if __name__ == "__main__":
    main()
