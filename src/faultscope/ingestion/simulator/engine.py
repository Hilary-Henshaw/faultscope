"""Machine simulator — generates realistic sensor readings.

Each :class:`MachineSimulator` models a single industrial machine
through its lifecycle.  At each call to :meth:`MachineSimulator.next_reading`
the simulator:

1. Advances the internal cycle counter.
2. Samples the :class:`DegradationCurve` to obtain a wear level.
3. Generates nominally-distributed sensor values with Gaussian noise.
4. Applies a wear-proportional multiplier to sensors listed in
   ``MachineProfile.degradation_sensors``.
5. Returns a :class:`~faultscope.common.kafka.schemas.SensorReading`.

Three machine profiles are defined: ``turbofan``, ``pump``, and
``compressor``.  The turbofan sensor set mirrors the 21-sensor layout
of the NASA C-MAPSS dataset so that simulated and real data share the
same schema downstream.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import numpy as np
import structlog

from faultscope.common.kafka.schemas import SensorReading
from faultscope.ingestion.simulator.failure_modes import (
    DegradationCurve,
    DegradationPattern,
)

log: structlog.BoundLogger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Operational setting ranges (shared across all profiles)
# ---------------------------------------------------------------------------
_OP_SETTING_RANGES: dict[str, tuple[float, float]] = {
    "op_setting_1": (-0.0010, 0.0010),
    "op_setting_2": (-0.0006, 0.0006),
    "op_setting_3": (99.0, 101.0),
}

# ---------------------------------------------------------------------------
# Turbofan sensor specification
# ---------------------------------------------------------------------------
_TURBOFAN_NOMINAL: dict[str, tuple[float, float]] = {
    "fan_inlet_temp": (518.67, 518.67),
    "lpc_outlet_temp": (642.0, 645.0),
    "hpc_outlet_temp": (1585.0, 1600.0),
    "lpt_outlet_temp": (1400.0, 1420.0),
    "total_pressure_r2": (14.62, 14.62),
    "total_pressure_r3": (21.60, 21.61),
    "burner_pressure": (554.0, 554.5),
    "lpt_pressure": (47.20, 47.25),
    "bleed_enthalpy": (521.0, 522.0),
    "engine_speed_physical": (2388.0, 2388.0),
    "engine_speed_corrected": (9046.0, 9047.0),
    "fan_speed_ratio": (1.30, 1.30),
    "corrected_fan_speed": (47.47, 47.48),
    "duct_pressure_ratio": (8.416, 8.416),
    "hpc_efficiency": (0.03, 0.04),
    "hpc_flow_balance": (392.0, 392.0),
    "bypass_ratio": (2388.0, 2388.0),
    "burner_fuel_ratio": (0.0218, 0.0219),
    "lpt_efficiency": (100.0, 100.0),
    "bleed_flow": (39.06, 39.08),
    "hpb_bleed_coolant": (23.419, 23.419),
}

_TURBOFAN_NOISE: dict[str, float] = {
    "fan_inlet_temp": 0.0,
    "lpc_outlet_temp": 0.50,
    "hpc_outlet_temp": 3.00,
    "lpt_outlet_temp": 3.00,
    "total_pressure_r2": 0.002,
    "total_pressure_r3": 0.002,
    "burner_pressure": 0.15,
    "lpt_pressure": 0.02,
    "bleed_enthalpy": 0.30,
    "engine_speed_physical": 0.0,
    "engine_speed_corrected": 1.00,
    "fan_speed_ratio": 0.0,
    "corrected_fan_speed": 0.01,
    "duct_pressure_ratio": 0.0,
    "hpc_efficiency": 0.003,
    "hpc_flow_balance": 0.0,
    "bypass_ratio": 0.0,
    "burner_fuel_ratio": 0.0001,
    "lpt_efficiency": 0.0,
    "bleed_flow": 0.01,
    "hpb_bleed_coolant": 0.0,
}

_TURBOFAN_DEGRADE: list[str] = [
    "lpc_outlet_temp",
    "hpc_outlet_temp",
    "lpt_outlet_temp",
    "burner_pressure",
    "lpt_pressure",
    "bleed_enthalpy",
    "hpc_efficiency",
    "burner_fuel_ratio",
    "bleed_flow",
]

# ---------------------------------------------------------------------------
# Pump sensor specification
# ---------------------------------------------------------------------------
_PUMP_NOMINAL: dict[str, tuple[float, float]] = {
    "temperature": (65.0, 75.0),
    "flow_rate": (450.0, 550.0),
    "inlet_pressure": (1.5, 2.5),
    "outlet_pressure": (8.0, 10.0),
    "vibration_x": (0.5, 1.5),
    "vibration_y": (0.5, 1.5),
    "power_consumption": (15.0, 20.0),
    "shaft_speed": (1450.0, 1550.0),
    "bearing_temp": (45.0, 55.0),
}

_PUMP_NOISE: dict[str, float] = {
    "temperature": 0.5,
    "flow_rate": 5.0,
    "inlet_pressure": 0.05,
    "outlet_pressure": 0.10,
    "vibration_x": 0.10,
    "vibration_y": 0.10,
    "power_consumption": 0.3,
    "shaft_speed": 5.0,
    "bearing_temp": 0.5,
}

_PUMP_DEGRADE: list[str] = [
    "temperature",
    "vibration_x",
    "vibration_y",
    "bearing_temp",
    "power_consumption",
]

# ---------------------------------------------------------------------------
# Compressor sensor specification
# ---------------------------------------------------------------------------
_COMPRESSOR_NOMINAL: dict[str, tuple[float, float]] = {
    "inlet_temp": (20.0, 25.0),
    "outlet_temp": (180.0, 220.0),
    "pressure_ratio": (7.5, 8.5),
    "mass_flow": (12.0, 14.0),
    "shaft_speed": (18000.0, 22000.0),
    "vibration_rms": (1.0, 3.0),
    "bearing_temp": (60.0, 80.0),
    "oil_temp": (50.0, 70.0),
    "surge_margin": (20.0, 30.0),
}

_COMPRESSOR_NOISE: dict[str, float] = {
    "inlet_temp": 0.2,
    "outlet_temp": 2.0,
    "pressure_ratio": 0.05,
    "mass_flow": 0.1,
    "shaft_speed": 50.0,
    "vibration_rms": 0.15,
    "bearing_temp": 0.5,
    "oil_temp": 0.5,
    "surge_margin": 0.5,
}

_COMPRESSOR_DEGRADE: list[str] = [
    "outlet_temp",
    "bearing_temp",
    "oil_temp",
    "vibration_rms",
    "surge_margin",
]

# ---------------------------------------------------------------------------
# Degradation direction: +1 means value increases with wear,
#                        -1 means value decreases.
# ---------------------------------------------------------------------------
_DEGRADE_DIRECTION: dict[str, float] = {
    # turbofan
    "lpc_outlet_temp": +1.0,
    "hpc_outlet_temp": +1.0,
    "lpt_outlet_temp": +1.0,
    "burner_pressure": -1.0,
    "lpt_pressure": -1.0,
    "bleed_enthalpy": +1.0,
    "hpc_efficiency": -1.0,
    "burner_fuel_ratio": +1.0,
    "bleed_flow": -1.0,
    # pump
    "temperature": +1.0,
    "vibration_x": +1.0,
    "vibration_y": +1.0,
    "bearing_temp": +1.0,
    "power_consumption": +1.0,
    # compressor
    "outlet_temp": +1.0,
    "vibration_rms": +1.0,
    "oil_temp": +1.0,
    "surge_margin": -1.0,
}

# Magnitude of degradation effect as fraction of nominal midpoint.
_DEGRADE_MAGNITUDE: float = 0.25


@dataclasses.dataclass(frozen=True)
class MachineProfile:
    """Defines sensor characteristics for a single machine type.

    Attributes
    ----------
    machine_type:
        Short name used as a prefix/label (``turbofan``, ``pump``,
        or ``compressor``).
    sensor_names:
        Ordered list of sensor identifiers produced by this profile.
    nominal_ranges:
        ``{sensor_name: (low, high)}`` — the healthy operating band.
        Simulated values are drawn from this interval at cycle 0.
    noise_scales:
        ``{sensor_name: sigma}`` — standard deviation of Gaussian
        noise added to each reading.
    degradation_sensors:
        Subset of ``sensor_names`` whose readings shift meaningfully
        as the machine wears.
    """

    machine_type: str
    sensor_names: list[str]
    nominal_ranges: dict[str, tuple[float, float]]
    noise_scales: dict[str, float]
    degradation_sensors: list[str]


def _make_turbofan_profile() -> MachineProfile:
    """Build the turbofan :class:`MachineProfile`."""
    return MachineProfile(
        machine_type="turbofan",
        sensor_names=list(_TURBOFAN_NOMINAL.keys()),
        nominal_ranges=_TURBOFAN_NOMINAL,
        noise_scales=_TURBOFAN_NOISE,
        degradation_sensors=_TURBOFAN_DEGRADE,
    )


def _make_pump_profile() -> MachineProfile:
    """Build the pump :class:`MachineProfile`."""
    return MachineProfile(
        machine_type="pump",
        sensor_names=list(_PUMP_NOMINAL.keys()),
        nominal_ranges=_PUMP_NOMINAL,
        noise_scales=_PUMP_NOISE,
        degradation_sensors=_PUMP_DEGRADE,
    )


def _make_compressor_profile() -> MachineProfile:
    """Build the compressor :class:`MachineProfile`."""
    return MachineProfile(
        machine_type="compressor",
        sensor_names=list(_COMPRESSOR_NOMINAL.keys()),
        nominal_ranges=_COMPRESSOR_NOMINAL,
        noise_scales=_COMPRESSOR_NOISE,
        degradation_sensors=_COMPRESSOR_DEGRADE,
    )


#: Registry of built-in profiles keyed by ``machine_type``.
PROFILES: dict[str, MachineProfile] = {
    "turbofan": _make_turbofan_profile(),
    "pump": _make_pump_profile(),
    "compressor": _make_compressor_profile(),
}


class MachineSimulator:
    """Simulates a single machine's sensor data over its lifecycle.

    Generates a stream of
    :class:`~faultscope.common.kafka.schemas.SensorReading`
    objects with realistic sensor correlations and configurable
    degradation patterns.  When the machine reaches end-of-life
    :meth:`next_reading` raises :exc:`StopIteration`.

    Parameters
    ----------
    machine_id:
        Unique identifier, used as the Kafka partition key and as
        ``SensorReading.machine_id``.
    profile:
        Sensor layout and nominal operating ranges.
    total_cycles:
        How many cycles the machine lives before it is considered
        failed.
    pattern:
        Which degradation curve shape to apply.
    rng:
        NumPy random generator for noise and degradation sampling.

    Raises
    ------
    ValueError
        If ``total_cycles < 1``.
    """

    def __init__(
        self,
        machine_id: str,
        profile: MachineProfile,
        total_cycles: int,
        pattern: DegradationPattern,
        rng: np.random.Generator,
    ) -> None:
        if total_cycles < 1:
            raise ValueError(f"total_cycles must be >= 1, got {total_cycles}")
        self._machine_id = machine_id
        self._profile = profile
        self._total_cycles = total_cycles
        self._rng = rng
        self._cycle: int = 0
        self._curve = DegradationCurve(pattern, total_cycles, rng)

        # Freeze the nominal midpoints once; noise is added per-cycle.
        self._nominal: dict[str, float] = {
            s: (lo + hi) / 2.0
            for s, (lo, hi) in profile.nominal_ranges.items()
        }
        log.info(
            "machine_simulator.created",
            machine_id=machine_id,
            machine_type=profile.machine_type,
            total_cycles=total_cycles,
            pattern=pattern.value,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def machine_id(self) -> str:
        """Unique machine identifier."""
        return self._machine_id

    @property
    def is_alive(self) -> bool:
        """``True`` while the machine has cycles remaining."""
        return self._cycle < self._total_cycles

    @property
    def current_rul(self) -> int:
        """Remaining cycles until failure (0 once failed)."""
        return max(0, self._total_cycles - self._cycle)

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def next_reading(self) -> SensorReading:
        """Advance one cycle and return the sensor reading.

        Returns
        -------
        SensorReading
            A fully populated reading for the current cycle.

        Raises
        ------
        StopIteration
            When the machine has reached or passed ``total_cycles``.
        """
        if not self.is_alive:
            raise StopIteration(
                f"Machine {self._machine_id} has reached end-of-life"
                f" at cycle {self._cycle}"
            )

        wear: float = self._curve.sample(self._cycle)

        readings: dict[str, float] = {}
        for sensor in self._profile.sensor_names:
            base = self._nominal[sensor]
            sigma = self._profile.noise_scales.get(sensor, 0.0)
            noise = float(self._rng.normal(0.0, sigma)) if sigma > 0 else 0.0
            value = base + noise

            if sensor in self._profile.degradation_sensors:
                direction = _DEGRADE_DIRECTION.get(sensor, +1.0)
                shift = direction * wear * _DEGRADE_MAGNITUDE * abs(base)
                value += shift

            readings[sensor] = round(value, 6)

        operational: dict[str, float] = {}
        for key, (lo, hi) in _OP_SETTING_RANGES.items():
            midpoint = (lo + hi) / 2.0
            half = (hi - lo) / 2.0
            operational[key] = round(
                float(self._rng.uniform(midpoint - half, midpoint + half)),
                6,
            )

        reading = SensorReading(
            machine_id=self._machine_id,
            recorded_at=datetime.now(tz=UTC),
            cycle=self._cycle,
            readings=readings,
            operational=operational,
        )

        log.debug(
            "machine_simulator.reading",
            machine_id=self._machine_id,
            cycle=self._cycle,
            wear=round(wear, 4),
            rul=self.current_rul,
        )

        self._cycle += 1
        return reading
