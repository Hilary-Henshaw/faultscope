"""Basic simulation example.

Demonstrates using MachineSimulator to generate synthetic sensor readings
and publishing them to Kafka via EventPublisher.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

from faultscope.common.config import FaultScopeSettings
from faultscope.common.kafka.producer import EventPublisher
from faultscope.common.kafka.schemas import SensorReading
from faultscope.common.logging import configure_logging, get_logger
from faultscope.ingestion.simulator.engine import (
    PROFILES,
    MachineSimulator,
)

logger = get_logger(__name__)


def _build_machines(
    machine_type: str,
    count: int,
) -> list[MachineSimulator]:
    """Create simulators for the requested machine type."""
    profile = PROFILES[machine_type]
    return [
        MachineSimulator(
            machine_id=f"example-{machine_type}-{i:03d}",
            profile=profile,
        )
        for i in range(1, count + 1)
    ]


async def run(
    machine_type: str,
    count: int,
    interval_ms: int,
    duration_s: float,
) -> None:
    """Run the simulation loop."""
    configure_logging(level="INFO", fmt="console")
    settings = FaultScopeSettings()

    publisher = EventPublisher(settings.kafka)
    await publisher.start()
    logger.info(
        "simulator_started",
        machines=count,
        type=machine_type,
    )

    machines = _build_machines(machine_type, count)
    start = time.monotonic()
    total_published = 0

    try:
        while True:
            elapsed = time.monotonic() - start
            if duration_s > 0 and elapsed >= duration_s:
                break

            for idx, sim in enumerate(machines):
                if not sim.is_alive:
                    # Replace failed machine with a fresh one
                    machines[idx] = MachineSimulator(
                        machine_id=f"example-{machine_type}-{idx + 1:03d}"
                        f"-r{int(elapsed):04d}",
                        profile=PROFILES[machine_type],
                    )
                    logger.info(
                        "machine_replaced",
                        machine_id=sim.machine_id,
                        elapsed_s=round(elapsed, 1),
                    )
                    sim = machines[idx]

                raw: dict[str, Any] = sim.next_reading()
                reading = SensorReading(
                    machine_id=raw["machine_id"],
                    sensor_readings=raw["sensor_readings"],
                    operational_settings=raw.get("operational_settings"),
                    cycle=raw.get("cycle"),
                )

                await publisher.publish(
                    topic=settings.kafka.topic_readings,
                    key=reading.machine_id,
                    value=reading,
                )
                total_published += 1
                logger.debug(
                    "reading_published",
                    machine_id=reading.machine_id,
                    rul=sim.current_rul,
                    sensors=len(reading.sensor_readings),
                )

            await asyncio.sleep(interval_ms / 1000)

    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await publisher.stop()
        logger.info(
            "simulation_complete",
            total_published=total_published,
            elapsed_s=round(time.monotonic() - start, 1),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FaultScope basic simulation example",
    )
    parser.add_argument(
        "--machine-type",
        choices=list(PROFILES.keys()),
        default="turbofan",
        help="Machine profile to simulate",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="Number of machines to simulate",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=500,
        help="Milliseconds between readings per cycle",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=30.0,
        help="Simulation duration in seconds (0 = run indefinitely)",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            machine_type=args.machine_type,
            count=args.count,
            interval_ms=args.interval_ms,
            duration_s=args.duration_s,
        )
    )


if __name__ == "__main__":
    main()
