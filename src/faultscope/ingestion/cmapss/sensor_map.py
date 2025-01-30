"""C-MAPSS column name definitions and sensor classification.

This module is the single source of truth for how the 26-column
NASA C-MAPSS text files map to human-readable names used throughout
the FaultScope ingestion pipeline.

References
----------
A. Saxena, K. Goebel, D. Simon, and N. Eklund,
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure
Simulation", *PHM '08*, 2008.
"""

from __future__ import annotations

#: All 26 column names in the order they appear in the raw text files.
#: Columns 1-2 are identifiers, 3-5 are operational settings, 6-26
#: are the 21 sensor measurements.
CMAPSS_SENSOR_COLUMNS: list[str] = [
    "engine_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    "fan_inlet_temp",  # s1  – T2   (°R)
    "lpc_outlet_temp",  # s2  – T24  (°R)
    "hpc_outlet_temp",  # s3  – T30  (°R)
    "lpt_outlet_temp",  # s4  – T50  (°R)
    "total_pressure_r2",  # s5  – P2   (psia)
    "total_pressure_r3",  # s6  – P15  (psia)
    "burner_pressure",  # s7  – P30  (psia)
    "lpt_pressure",  # s8  – Nf   (rpm) — note NASA labelling
    "bleed_enthalpy",  # s9  – Nc   (rpm)
    "engine_speed_physical",  # s10 – epr  (ratio)
    "engine_speed_corrected",  # s11 – Ps30 (psia)
    "fan_speed_ratio",  # s12 – phi  (pps/psi)
    "corrected_fan_speed",  # s13 – NRf  (rpm)
    "duct_pressure_ratio",  # s14 – NRc  (rpm)
    "hpc_efficiency",  # s15 – BPR  (dimensionless)
    "hpc_flow_balance",  # s16 – farB (dimensionless)
    "bypass_ratio",  # s17 – htBleed (dimensionless)
    "burner_fuel_ratio",  # s18 – Nf_dmd (rpm)
    "lpt_efficiency",  # s19 – PCNfR_dmd (rpm)
    "bleed_flow",  # s20 – W31  (lbm/s)
    "hpb_bleed_coolant",  # s21 – W32  (lbm/s)
]

#: Sensors whose variance is near-zero across all operating conditions
#: in the standard FD00x datasets.  These are typically excluded from
#: downstream feature engineering and ML model inputs.
LOW_VARIANCE_SENSORS: frozenset[str] = frozenset(
    {
        "engine_speed_physical",
        "hpc_flow_balance",
        "duct_pressure_ratio",
        "hpb_bleed_coolant",
    }
)

#: The three operational setting columns.
OPERATIONAL_COLUMNS: list[str] = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
]

#: The 21 sensor measurement columns (excludes identifiers and
#: operational settings).
SENSOR_COLUMNS: list[str] = [
    c
    for c in CMAPSS_SENSOR_COLUMNS
    if c not in ("engine_id", "cycle", *OPERATIONAL_COLUMNS)
]
