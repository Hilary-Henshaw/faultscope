#!/usr/bin/env python3
"""Check health of all FaultScope services.

Performs HTTP liveness probes against all five application services,
then checks Kafka broker reachability and TimescaleDB connectivity.
Prints a colour-coded status table and exits with code 1 if any
check fails.

Usage::

    python scripts/health_check.py
    python scripts/health_check.py --timeout=10
    python scripts/health_check.py --no-colour
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"


def _ok(text: str, colour: bool = True) -> str:
    return f"{_GREEN}{text}{_RESET}" if colour else text


def _fail(text: str, colour: bool = True) -> str:
    return f"{_RED}{text}{_RESET}" if colour else text


def _warn(text: str, colour: bool = True) -> str:
    return f"{_YELLOW}{text}{_RESET}" if colour else text


def _bold(text: str, colour: bool = True) -> str:
    return f"{_BOLD}{text}{_RESET}" if colour else text


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Status(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    WARN = "WARN"


@dataclass
class CheckResult:
    """Result of a single health check."""

    service: str
    status: Status
    detail: str = ""
    latency_ms: float = 0.0
    extra: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTTP health checks
# ---------------------------------------------------------------------------

_HTTP_SERVICES: list[tuple[str, str]] = [
    ("Inference API", "http://localhost:8000/health"),
    ("Alerting API", "http://localhost:8001/health"),
    ("Streaming Processor", "http://localhost:8002/health"),
    ("Grafana", "http://localhost:3000/api/health"),
    ("MLflow", "http://localhost:5000/health"),
]


async def _check_http(
    service: str,
    url: str,
    timeout: float,
    client: httpx.AsyncClient,
) -> CheckResult:
    """Probe a single HTTP health endpoint."""
    t0 = time.perf_counter()
    try:
        resp = await client.get(url, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            # Try to surface any status field from the JSON body
            extra: dict[str, str] = {}
            try:
                body = resp.json()
                if isinstance(body, dict):
                    sv = body.get("status") or body.get("database")
                    if sv:
                        extra["body_status"] = str(sv)
            except Exception:  # noqa: BLE001
                pass
            return CheckResult(
                service=service,
                status=Status.UP,
                detail=f"HTTP {resp.status_code}",
                latency_ms=latency,
                extra=extra,
            )
        return CheckResult(
            service=service,
            status=Status.WARN,
            detail=f"HTTP {resp.status_code}",
            latency_ms=latency,
        )
    except httpx.ConnectError:
        latency = (time.perf_counter() - t0) * 1000
        return CheckResult(
            service=service,
            status=Status.DOWN,
            detail="Connection refused",
            latency_ms=latency,
        )
    except httpx.TimeoutException:
        latency = (time.perf_counter() - t0) * 1000
        return CheckResult(
            service=service,
            status=Status.DOWN,
            detail=f"Timed out after {timeout:.0f}s",
            latency_ms=latency,
        )
    except Exception as exc:  # noqa: BLE001
        latency = (time.perf_counter() - t0) * 1000
        return CheckResult(
            service=service,
            status=Status.DOWN,
            detail=str(exc)[:60],
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Kafka check
# ---------------------------------------------------------------------------

async def _check_kafka(
    bootstrap_servers: str,
    timeout: float,
) -> CheckResult:
    """Attempt to fetch Kafka cluster metadata."""
    t0 = time.perf_counter()
    try:
        from aiokafka.admin import AIOKafkaAdminClient

        admin = AIOKafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            request_timeout_ms=int(timeout * 1000),
        )
        await admin.start()
        try:
            topics = await admin.list_topics()
        finally:
            await admin.close()
        latency = (time.perf_counter() - t0) * 1000
        faultscope_topics = [
            t for t in (topics or []) if "faultscope" in t
        ]
        return CheckResult(
            service="Kafka",
            status=Status.UP,
            detail=f"{len(faultscope_topics)} faultscope topics",
            latency_ms=latency,
        )
    except ImportError:
        return CheckResult(
            service="Kafka",
            status=Status.WARN,
            detail="aiokafka not installed; skipped",
        )
    except Exception as exc:  # noqa: BLE001
        latency = (time.perf_counter() - t0) * 1000
        return CheckResult(
            service="Kafka",
            status=Status.DOWN,
            detail=str(exc)[:60],
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# TimescaleDB check
# ---------------------------------------------------------------------------

async def _check_timescaledb(
    host: str,
    port: int,
    dbname: str,
    user: str,
    password: str,
    timeout: float,
) -> CheckResult:
    """Connect to TimescaleDB and query basic metadata."""
    t0 = time.perf_counter()
    try:
        import asyncpg

        conn: asyncpg.Connection = await asyncio.wait_for(  # type: ignore[type-arg]
            asyncpg.connect(
                host=host,
                port=port,
                database=dbname,
                user=user,
                password=password,
            ),
            timeout=timeout,
        )
        try:
            row = await conn.fetchrow(
                "SELECT count(*) AS n FROM machines"
            )
            latency = (time.perf_counter() - t0) * 1000
            machines = int(row["n"]) if row else 0
        finally:
            await conn.close()
        return CheckResult(
            service="TimescaleDB",
            status=Status.UP,
            detail=f"{machines} machines registered",
            latency_ms=latency,
        )
    except ImportError:
        return CheckResult(
            service="TimescaleDB",
            status=Status.WARN,
            detail="asyncpg not installed; skipped",
        )
    except asyncio.TimeoutError:
        latency = (time.perf_counter() - t0) * 1000
        return CheckResult(
            service="TimescaleDB",
            status=Status.DOWN,
            detail=f"Timed out after {timeout:.0f}s",
            latency_ms=latency,
        )
    except Exception as exc:  # noqa: BLE001
        latency = (time.perf_counter() - t0) * 1000
        return CheckResult(
            service="TimescaleDB",
            status=Status.DOWN,
            detail=str(exc)[:60],
            latency_ms=latency,
        )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_COL_SERVICE = 22
_COL_STATUS = 8
_COL_LATENCY = 10
_COL_DETAIL = 34


def _render_table(results: list[CheckResult], colour: bool) -> str:
    header = (
        f"{'Service':<{_COL_SERVICE}} "
        f"{'Status':<{_COL_STATUS}} "
        f"{'Latency':>{_COL_LATENCY}} "
        f"{'Detail':<{_COL_DETAIL}}"
    )
    sep = "-" * (
        _COL_SERVICE + _COL_STATUS + _COL_LATENCY + _COL_DETAIL + 3
    )
    lines: list[str] = [
        _bold("FaultScope Health Check", colour),
        _bold("=" * len(sep), colour),
        header,
        sep,
    ]
    for r in results:
        if r.status == Status.UP:
            status_str = _ok(f"{'UP':<{_COL_STATUS}}", colour)
        elif r.status == Status.WARN:
            status_str = _warn(f"{'WARN':<{_COL_STATUS}}", colour)
        else:
            status_str = _fail(f"{'DOWN':<{_COL_STATUS}}", colour)

        latency = (
            f"{r.latency_ms:6.0f} ms"
            if r.latency_ms > 0
            else f"{'—':>8}"
        )
        lines.append(
            f"{r.service:<{_COL_SERVICE}} "
            f"{status_str} "
            f"{latency:>{_COL_LATENCY}} "
            f"{r.detail:<{_COL_DETAIL}}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def run_checks(timeout: float, colour: bool) -> list[CheckResult]:
    """Execute all health checks concurrently."""
    # Load env settings without crashing if not configured
    kafka_servers = "localhost:9092"
    db_host = "localhost"
    db_port = 5432
    db_name = "faultscope"
    db_user = "faultscope"
    db_pass = "changeme_in_production"

    try:
        from faultscope.common.config import (
            DatabaseSettings,
            KafkaSettings,
        )

        ks = KafkaSettings()  # type: ignore[call-arg]
        kafka_servers = ks.bootstrap_servers
        ds = DatabaseSettings()  # type: ignore[call-arg]
        db_host = ds.host
        db_port = ds.port
        db_name = ds.name
        db_user = ds.user
        db_pass = ds.password.get_secret_value()
    except Exception:  # noqa: BLE001
        pass  # Use defaults; still attempt the check

    async with httpx.AsyncClient() as client:
        http_tasks = [
            _check_http(svc, url, timeout, client)
            for svc, url in _HTTP_SERVICES
        ]
        kafka_task = _check_kafka(kafka_servers, timeout)
        db_task = _check_timescaledb(
            db_host, db_port, db_name,
            db_user, db_pass, timeout,
        )
        results = await asyncio.gather(
            *http_tasks, kafka_task, db_task
        )

    return list(results)


async def main(timeout: float, colour: bool) -> int:
    """Run all checks, print table, return exit code."""
    results = await run_checks(timeout=timeout, colour=colour)
    print(f"\n{_render_table(results, colour)}\n")

    n_down = sum(1 for r in results if r.status == Status.DOWN)
    n_warn = sum(1 for r in results if r.status == Status.WARN)
    n_up = sum(1 for r in results if r.status == Status.UP)

    summary = (
        f"Services: {n_up} up, "
        f"{n_warn} warning, "
        f"{n_down} down"
    )
    if n_down > 0:
        print(_fail(summary, colour))
    elif n_warn > 0:
        print(_warn(summary, colour))
    else:
        print(_ok(summary, colour))
    print()

    return 1 if n_down > 0 else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check health of all FaultScope services. "
            "Exits with code 1 if any service is DOWN."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP / connection timeout in seconds per check",
    )
    parser.add_argument(
        "--no-colour",
        action="store_true",
        default=False,
        help="Disable ANSI colour output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    exit_code = asyncio.run(
        main(timeout=args.timeout, colour=not args.no_colour)
    )
    sys.exit(exit_code)
