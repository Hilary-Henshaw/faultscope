"""CLI entry point for the FaultScope model training pipeline.

Usage::

    python -m faultscope.training --dataset-version=v1

    # Or via the installed script:
    faultscope-train --dataset-version=v1
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import structlog

from faultscope.common.logging import configure_logging
from faultscope.training.config import TrainingConfig
from faultscope.training.pipeline import TrainingOrchestrator

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="faultscope-train",
        description="Run the FaultScope model training pipeline.",
    )
    parser.add_argument(
        "--dataset-version",
        default="v1",
        metavar="VERSION",
        help=(
            "Dataset version tag to load from the feature store (default: v1)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO).",
    )
    parser.add_argument(
        "--log-format",
        default="json",
        choices=["json", "console"],
        help="Log output format (default: json).",
    )
    return parser


async def _async_main(dataset_version: str) -> dict[str, str]:
    """Run the training orchestrator asynchronously."""
    config = TrainingConfig()
    orchestrator = TrainingOrchestrator(config)
    return await orchestrator.run(dataset_version=dataset_version)


def main() -> None:
    """CLI entry point: ``python -m faultscope.training``."""
    parser = _build_parser()
    args = parser.parse_args()

    configure_logging(level=args.log_level, fmt=args.log_format)

    log.info(
        "training_cli_started",
        dataset_version=args.dataset_version,
    )

    try:
        run_ids = asyncio.run(_async_main(args.dataset_version))
    except Exception as exc:
        log.error("training_pipeline_failed", error=str(exc), exc_info=True)
        sys.exit(1)

    log.info(
        "training_cli_finished",
        rul_run_id=run_ids["rul_run_id"],
        health_run_id=run_ids["health_run_id"],
    )


if __name__ == "__main__":
    main()
