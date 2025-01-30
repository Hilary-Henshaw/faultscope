"""Entry point for the FaultScope ingestion service.

Invoke as::

    python -m faultscope.ingestion           # default: --mode simulate
    python -m faultscope.ingestion --mode simulate
    python -m faultscope.ingestion --mode cmapss

Or via the installed console script::

    faultscope-ingest [--mode simulate|cmapss]

Modes
-----
simulate
    Spawns ``N``
    :class:`~faultscope.ingestion.simulator.engine.MachineSimulator`
    instances (configured via ``FAULTSCOPE_INGEST_MACHINES``) and emits
    one :class:`~faultscope.common.kafka.schemas.SensorReading` per
    machine per ``FAULTSCOPE_INGEST_INTERVAL_S`` seconds.  Machines
    that reach end-of-life are automatically replaced with fresh ones
    using a new random degradation pattern.

cmapss
    Loads the NASA C-MAPSS dataset from ``FAULTSCOPE_CMAPSS_DATA_PATH``
    and streams every row to Kafka at the configured interval.  Requires
    ``FAULTSCOPE_ENABLE_CMAPSS=true`` as a safety guard.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import signal
import sys

import numpy as np
import structlog

from faultscope.common.logging import configure_logging, get_logger
from faultscope.ingestion.cmapss.loader import CmapssLoader
from faultscope.ingestion.config import IngestionConfig
from faultscope.ingestion.publisher import SensorPublisher
from faultscope.ingestion.simulator.engine import PROFILES, MachineSimulator
from faultscope.ingestion.simulator.failure_modes import DegradationPattern

log: structlog.stdlib.BoundLogger = get_logger(__name__)

#: Minimum and maximum total lifecycle cycles assigned to a new machine.
_MIN_CYCLES: int = 150
_MAX_CYCLES: int = 500

#: Degradation patterns available for random assignment.
_ALL_PATTERNS: list[DegradationPattern] = list(DegradationPattern)

#: Machine types available for random assignment.
_ALL_TYPES: list[str] = list(PROFILES.keys())


def _new_simulator(
    machine_id: str,
    rng: np.random.Generator,
) -> MachineSimulator:
    """Create a fresh :class:`MachineSimulator` with random parameters.

    Parameters
    ----------
    machine_id:
        Persistent identifier for this slot (survives restarts).
    rng:
        Shared generator; each call advances its state so successive
        machines are distinct but the overall sequence is reproducible.

    Returns
    -------
    MachineSimulator
        A ready-to-run simulator.
    """
    machine_type = str(rng.choice(_ALL_TYPES))  # type: ignore[arg-type]
    pattern_idx = int(rng.integers(0, len(_ALL_PATTERNS)))
    pattern = _ALL_PATTERNS[pattern_idx]
    total_cycles = int(rng.integers(_MIN_CYCLES, _MAX_CYCLES + 1))
    profile = PROFILES[machine_type]

    log.info(
        "ingestion.new_simulator",
        machine_id=machine_id,
        machine_type=machine_type,
        pattern=pattern.value,
        total_cycles=total_cycles,
    )
    return MachineSimulator(
        machine_id=machine_id,
        profile=profile,
        total_cycles=total_cycles,
        pattern=pattern,
        rng=rng,
    )


async def run_simulation(config: IngestionConfig) -> None:
    """Spawn N MachineSimulators and publish readings in a loop.

    Each machine is assigned a slot ID (``SIM-001`` … ``SIM-NNN``).
    When a machine reaches end-of-life its slot is repopulated with a
    new simulator so that the stream is continuous.

    The coroutine runs until it receives a :exc:`asyncio.CancelledError`
    (e.g. from a SIGINT/SIGTERM handler installed in :func:`main`).

    Parameters
    ----------
    config:
        Validated ingestion configuration.

    Raises
    ------
    asyncio.CancelledError
        Propagated to the caller when the task is cancelled.
    """
    rng = np.random.default_rng(config.degradation_seed)

    simulators: list[MachineSimulator] = [
        _new_simulator(f"SIM-{i + 1:03d}", rng)
        for i in range(config.num_machines)
    ]

    async with SensorPublisher(
        bootstrap_servers=config.kafka_bootstrap_servers,
        topic=config.topic_sensor_readings,
    ) as publisher:
        log.info(
            "ingestion.simulation_started",
            num_machines=config.num_machines,
            interval_s=config.emit_interval_s,
        )

        while True:
            for idx, sim in enumerate(simulators):
                if not sim.is_alive:
                    # Replace expired machine with a fresh one at same
                    # slot ID so downstream consumers see a restart.
                    simulators[idx] = _new_simulator(sim.machine_id, rng)
                    sim = simulators[idx]

                try:
                    reading = sim.next_reading()
                except StopIteration:
                    # Rare race: simulator expired between the is_alive
                    # check and next_reading(); replace and skip cycle.
                    simulators[idx] = _new_simulator(sim.machine_id, rng)
                    continue

                await publisher.send_reading(reading)

            await asyncio.sleep(config.emit_interval_s)


async def run_cmapss(config: IngestionConfig) -> None:
    """Stream C-MAPSS dataset rows to Kafka.

    Iterates over all configured sub-datasets in a round-robin loop,
    yielding one :class:`~faultscope.common.kafka.schemas.SensorReading`
    per row.  After the last row of the last dataset the loop restarts
    from the beginning so the service keeps running until cancelled.

    Parameters
    ----------
    config:
        Validated ingestion configuration.  ``enable_cmapss`` must be
        ``True`` and ``cmapss_data_path`` must point to a directory
        containing the raw ``.txt`` files.

    Raises
    ------
    ValueError
        If ``enable_cmapss`` is ``False`` in ``config``.
    FileNotFoundError
        If ``cmapss_data_path`` does not exist.
    asyncio.CancelledError
        Propagated to the caller when the task is cancelled.
    """
    if not config.enable_cmapss:
        raise ValueError(
            "run_cmapss called but FAULTSCOPE_ENABLE_CMAPSS is False. "
            "Set it to true or use --mode simulate."
        )

    loader = CmapssLoader(data_path=config.cmapss_data_path)

    async with SensorPublisher(
        bootstrap_servers=config.kafka_bootstrap_servers,
        topic=config.topic_sensor_readings,
    ) as publisher:
        log.info(
            "ingestion.cmapss_started",
            data_path=config.cmapss_data_path,
            interval_s=config.emit_interval_s,
        )

        # Infinite loop: cycle through datasets repeatedly.
        for dataset_id in itertools.cycle(
            ["FD001", "FD002", "FD003", "FD004"]
        ):
            log.info(
                "ingestion.cmapss_dataset",
                dataset_id=dataset_id,
            )
            for reading in loader.iter_readings(dataset_id):
                await publisher.send_reading(reading)
                await asyncio.sleep(config.emit_interval_s)


def _install_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    main_task: asyncio.Task[None],
) -> None:
    """Install SIGINT / SIGTERM handlers that cancel ``main_task``.

    Parameters
    ----------
    loop:
        The running event loop.
    main_task:
        The top-level asyncio task to cancel on signal receipt.
    """

    def _handle_signal(signame: str) -> None:
        log.info("ingestion.signal_received", signal=signame)
        main_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal, sig.name)


async def _run(mode: str, config: IngestionConfig) -> None:
    """Dispatch to the correct run-function based on ``mode``.

    Parameters
    ----------
    mode:
        Either ``"simulate"`` or ``"cmapss"``.
    config:
        Loaded and validated ingestion configuration.

    Raises
    ------
    ValueError
        If ``mode`` is not ``"simulate"`` or ``"cmapss"``.
    """
    if mode == "simulate":
        await run_simulation(config)
    elif mode == "cmapss":
        await run_cmapss(config)
    else:
        raise ValueError(
            f"Unknown mode {mode!r}. Expected 'simulate' or 'cmapss'."
        )


def main() -> None:
    """Parse CLI arguments, load config, and run the ingestion service.

    This function is registered as the ``faultscope-ingest`` console
    script in ``pyproject.toml``.

    Exit codes
    ----------
    0
        Clean shutdown (SIGINT / SIGTERM received).
    1
        Fatal configuration or runtime error.
    """
    parser = argparse.ArgumentParser(
        prog="faultscope-ingest",
        description="FaultScope data ingestion service",
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "cmapss"],
        default="simulate",
        help=(
            "simulate: synthetic machine data (default). "
            "cmapss: stream NASA C-MAPSS dataset."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--log-format",
        default="json",
        choices=["json", "console"],
        help=(
            "Log renderer: 'json' for production, "
            "'console' for development (default: json)."
        ),
    )
    args = parser.parse_args()

    configure_logging(level=args.log_level, fmt=args.log_format)

    try:
        config = IngestionConfig()
    except Exception as exc:
        # Avoid crashing before structlog is fully initialised.
        log.error(
            "ingestion.config_error",
            error=str(exc),
        )
        sys.exit(1)

    log.info(
        "ingestion.startup",
        mode=args.mode,
        kafka_bootstrap_servers=config.kafka_bootstrap_servers,
        topic=config.topic_sensor_readings,
        num_machines=config.num_machines,
        emit_interval_s=config.emit_interval_s,
        enable_cmapss=config.enable_cmapss,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    main_task: asyncio.Task[None] = loop.create_task(_run(args.mode, config))
    _install_signal_handlers(loop, main_task)

    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        log.info("ingestion.shutdown")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        log.error("ingestion.fatal_error", error=str(exc))
        sys.exit(1)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
