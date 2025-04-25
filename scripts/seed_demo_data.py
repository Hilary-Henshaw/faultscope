#!/usr/bin/env python3
"""Seed FaultScope with demo machine data.

Connects to TimescaleDB and inserts five demo machines, then uses
:class:`~faultscope.ingestion.simulator.engine.MachineSimulator` to
generate synthetic sensor data and publishes every reading to the
``faultscope.sensors.readings`` Kafka topic.

Usage::

    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --machines=5 --cycles=200
    python scripts/seed_demo_data.py --cycles=50 --seed=123
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass

import asyncpg
import numpy as np

from faultscope.common.config import DatabaseSettings, KafkaSettings
from faultscope.common.logging import get_logger
from faultscope.ingestion.publisher import SensorPublisher
from faultscope.ingestion.simulator.engine import (
    PROFILES,
    MachineSimulator,
)
from faultscope.ingestion.simulator.failure_modes import (
    DegradationPattern,
)

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Demo machine definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DemoMachine:
    """Static description of a single demo machine."""

    machine_id: str
    machine_type: str
    location: str
    pattern: DegradationPattern


DEMO_MACHINES: list[DemoMachine] = [
    DemoMachine(
        machine_id="FAN-001",
        machine_type="turbofan",
        location="Plant-A / Bay-1",
        pattern=DegradationPattern.LINEAR,
    ),
    DemoMachine(
        machine_id="FAN-002",
        machine_type="turbofan",
        location="Plant-A / Bay-2",
        pattern=DegradationPattern.EXPONENTIAL,
    ),
    DemoMachine(
        machine_id="PUMP-001",
        machine_type="pump",
        location="Plant-B / Coolant-Line",
        pattern=DegradationPattern.STEP,
    ),
    DemoMachine(
        machine_id="COMP-001",
        machine_type="compressor",
        location="Plant-C / Compressor-Bay",
        pattern=DegradationPattern.OSCILLATING,
    ),
    DemoMachine(
        machine_id="MOTOR-001",
        machine_type="turbofan",
        location="Plant-A / Drive-Bay",
        pattern=DegradationPattern.LINEAR,
    ),
]


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

async def _ensure_machines(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    machines: list[DemoMachine],
) -> None:
    """Insert demo machine rows; skip if the machine_id already exists."""
    for machine in machines:
        machine_type = machine.machine_type
        # MOTOR-001 is typed as turbofan in the simulator but stored as
        # 'motor' in the DB to exercise that enum branch.
        if machine.machine_id.startswith("MOTOR"):
            machine_type = "motor"

        await conn.execute(
            """
            INSERT INTO machines
                (machine_id, machine_type, location, status)
            VALUES ($1, $2, $3, 'active')
            ON CONFLICT (machine_id) DO NOTHING
            """,
            machine.machine_id,
            machine_type,
            machine.location,
        )
    _log.info(
        "seed.machines_ensured",
        count=len(machines),
    )


async def _insert_reading(
    conn: asyncpg.Connection,  # type: ignore[type-arg]
    machine_id: str,
    cycle: int,
    readings: dict[str, float],
    operational: dict[str, float],
) -> None:
    """Write one sensor reading row to TimescaleDB."""
    import json
    from datetime import datetime, timezone

    await conn.execute(
        """
        INSERT INTO sensor_readings
            (recorded_at, machine_id, cycle, readings, operational)
        VALUES (NOW(), $1, $2, $3::jsonb, $4::jsonb)
        """,
        machine_id,
        cycle,
        json.dumps(readings),
        json.dumps(operational),
    )


# ---------------------------------------------------------------------------
# Simulation logic
# ---------------------------------------------------------------------------

async def _seed_machine(
    demo: DemoMachine,
    cycles: int,
    seed: int,
    db_dsn: str,
    publisher: SensorPublisher,
) -> dict[str, int | str]:
    """Run simulation for one machine and publish all readings.

    Returns a summary dict for the progress table.
    """
    rng = np.random.default_rng(seed)
    profile = PROFILES[demo.machine_type]
    simulator = MachineSimulator(
        machine_id=demo.machine_id,
        profile=profile,
        total_cycles=cycles,
        pattern=demo.pattern,
        rng=rng,
    )

    conn: asyncpg.Connection = await asyncpg.connect(dsn=db_dsn)  # type: ignore[type-arg]
    published = 0
    try:
        while simulator.is_alive:
            reading = simulator.next_reading()
            # Publish to Kafka
            await publisher.send_reading(reading)
            # Persist to TimescaleDB
            await _insert_reading(
                conn,
                machine_id=reading.machine_id,
                cycle=reading.cycle or 0,
                readings=reading.readings,
                operational=reading.operational,
            )
            published += 1
    finally:
        await conn.close()

    _log.info(
        "seed.machine_complete",
        machine_id=demo.machine_id,
        cycles_published=published,
    )
    return {
        "machine_id": demo.machine_id,
        "machine_type": demo.machine_type,
        "location": demo.location,
        "pattern": demo.pattern.value,
        "readings_published": published,
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

async def main(machines: int, cycles: int, seed: int) -> None:
    """Seed FaultScope with demo data."""
    db_settings = DatabaseSettings()  # type: ignore[call-arg]
    kafka_settings = KafkaSettings()  # type: ignore[call-arg]

    # Build asyncpg DSN from settings
    pwd = db_settings.password.get_secret_value()
    db_dsn = (
        f"postgresql://{db_settings.user}:{pwd}"
        f"@{db_settings.host}:{db_settings.port}"
        f"/{db_settings.name}"
    )

    selected = DEMO_MACHINES[:machines]
    print(
        f"\nFaultScope Demo Seed\n"
        f"{'=' * 45}\n"
        f"  Machines : {len(selected)}\n"
        f"  Cycles   : {cycles} per machine\n"
        f"  RNG seed : {seed}\n"
        f"  Kafka    : {kafka_settings.bootstrap_servers}\n"
        f"  DB host  : {db_settings.host}:{db_settings.port}\n"
    )

    # Register demo machines in TimescaleDB
    conn: asyncpg.Connection = await asyncpg.connect(dsn=db_dsn)  # type: ignore[type-arg]
    try:
        await _ensure_machines(conn, selected)
    finally:
        await conn.close()

    publisher = SensorPublisher(
        bootstrap_servers=kafka_settings.bootstrap_servers,
        topic=kafka_settings.topic_sensor_readings,
    )

    start = time.perf_counter()
    summaries: list[dict[str, int | str]] = []
    async with publisher:
        tasks = [
            _seed_machine(
                demo=demo,
                cycles=cycles,
                seed=seed + i,
                db_dsn=db_dsn,
                publisher=publisher,
            )
            for i, demo in enumerate(selected)
        ]
        summaries = await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - start
    total_readings = sum(
        int(s["readings_published"]) for s in summaries
    )

    print(f"\nSeed Complete  ({elapsed:.1f}s)\n{'=' * 55}")
    header = (
        f"{'Machine':<12} {'Type':<12} "
        f"{'Pattern':<14} {'Readings':>9}"
    )
    print(header)
    print("-" * 55)
    for s in summaries:
        print(
            f"{s['machine_id']:<12} {s['machine_type']:<12} "
            f"{s['pattern']:<14} {s['readings_published']:>9,}"
        )
    print("-" * 55)
    print(f"{'TOTAL':<39} {total_readings:>9,}")
    print(
        f"\nTopic : {kafka_settings.topic_sensor_readings}\n"
        f"Rate  : {total_readings / elapsed:,.0f} readings/sec\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed FaultScope demo machines and sensor data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--machines",
        type=int,
        default=5,
        choices=range(1, len(DEMO_MACHINES) + 1),
        metavar=f"1-{len(DEMO_MACHINES)}",
        help="Number of demo machines to seed (max 5)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=200,
        help="Sensor cycles to simulate per machine",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base NumPy RNG seed for reproducible data",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(
            main(
                machines=args.machines,
                cycles=args.cycles,
                seed=args.seed,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        print(f"\nError: {exc}", file=sys.stderr)
        _log.exception("seed.failed", error=str(exc))
        sys.exit(1)
