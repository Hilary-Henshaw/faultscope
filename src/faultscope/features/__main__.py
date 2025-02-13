"""CLI entry point for the offline feature pipeline.

Usage::

    python -m faultscope.features \\
        --start=2024-01-01T00:00:00Z \\
        --end=2024-03-31T23:59:59Z

    # Or via the installed script:
    faultscope-features --start=2024-01-01T00:00:00Z \\
                        --end=2024-03-31T23:59:59Z
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime

import structlog

from faultscope.common.logging import configure_logging
from faultscope.features.config import FeaturesConfig
from faultscope.features.pipeline import FeaturePipelineRunner

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def _parse_utc_datetime(value: str) -> datetime:
    """Parse an ISO-8601 string and return a UTC-aware datetime.

    Parameters
    ----------
    value:
        ISO-8601 datetime string, e.g. ``"2024-01-01T00:00:00Z"`` or
        ``"2024-01-01T00:00:00+00:00"``.

    Returns
    -------
    datetime
        Timezone-aware datetime in UTC.

    Raises
    ------
    argparse.ArgumentTypeError
        If the string cannot be parsed.
    """
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Cannot parse datetime '{value}'. "
        "Expected ISO-8601 format, e.g. 2024-01-01T00:00:00Z"
    )


def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="faultscope-features",
        description="Run the FaultScope offline feature pipeline.",
    )
    parser.add_argument(
        "--start",
        required=True,
        type=_parse_utc_datetime,
        metavar="DATETIME",
        help=(
            "Inclusive start of the extraction window, "
            "e.g. 2024-01-01T00:00:00Z"
        ),
    )
    parser.add_argument(
        "--end",
        required=True,
        type=_parse_utc_datetime,
        metavar="DATETIME",
        help=(
            "Inclusive end of the extraction window, e.g. 2024-03-31T23:59:59Z"
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


async def _async_main(start: datetime, end: datetime) -> None:
    """Run the pipeline asynchronously."""
    config = FeaturesConfig()
    runner = FeaturePipelineRunner(config)
    try:
        splits = await runner.run(start, end)
        for name, df in splits.items():
            log.info(
                "split_summary",
                split=name,
                n_rows=len(df),
                n_machines=df["machine_id"].nunique(),
            )
    finally:
        await runner.close()


def main() -> None:
    """CLI entry point: ``python -m faultscope.features``."""
    parser = _build_parser()
    args = parser.parse_args()

    configure_logging(level=args.log_level, fmt=args.log_format)

    if args.start >= args.end:
        log.error(
            "invalid_time_range",
            start=args.start.isoformat(),
            end=args.end.isoformat(),
        )
        sys.exit(1)

    log.info(
        "feature_pipeline_cli_started",
        start=args.start.isoformat(),
        end=args.end.isoformat(),
    )

    try:
        asyncio.run(_async_main(args.start, args.end))
    except Exception as exc:
        log.error("feature_pipeline_failed", error=str(exc), exc_info=True)
        sys.exit(1)

    log.info("feature_pipeline_cli_finished")


if __name__ == "__main__":
    main()
