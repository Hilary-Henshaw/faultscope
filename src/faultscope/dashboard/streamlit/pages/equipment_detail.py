"""Equipment detail page: deep-dive view for a single machine.

Displays RUL trend with confidence intervals, health probability
distribution, recent sensor readings, and machine-specific incidents.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st
import structlog

from faultscope.dashboard.streamlit.components.api_client import (
    fetch_incidents,
    fetch_machine_predictions,
    fetch_machines,
    fetch_sensor_readings,
)
from faultscope.dashboard.streamlit.components.charts import (
    health_distribution_chart,
    rul_trend_chart,
    sensor_trend_chart,
)
from faultscope.dashboard.streamlit.config import DashboardConfig

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_HEALTH_COLOURS: dict[str, str] = {
    "healthy": "#22c55e",
    "degrading": "#eab308",
    "critical": "#f97316",
    "imminent_failure": "#ef4444",
}

# Sensors shown in the readings grid (shown if present in the data)
_KEY_SENSORS = [
    "fan_speed",
    "core_speed",
    "total_temperature_hpc_outlet",
    "total_temperature_lpt_outlet",
    "pressure_ratio",
    "static_pressure_hpc_outlet",
    "fuel_flow_ratio",
    "corrected_fan_speed",
    "corrected_core_speed",
    "bypass_ratio",
    "bleed_enthalpy",
    "required_fan_speed",
    "high_pressure_turbine_coolant_bleed",
    "low_pressure_turbine_coolant_bleed",
    "vibration_x",
    "vibration_y",
    "temperature",
]


def _parse_dt(value: object) -> datetime | None:
    """Parse an ISO-8601 string to a UTC-aware datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        return None


def render_equipment_page(
    machine_id: str,
    config: DashboardConfig,
) -> None:
    """Render the detail view for a single machine.

    Layout
    ------
    1. Machine header: ID, type, status, commissioned date.
    2. RUL trend chart for the last 24 h with confidence intervals.
    3. Health probability horizontal bar chart.
    4. Sensor readings grid (most recent hour, key sensors only).
    5. Incident history table for this machine.

    Parameters
    ----------
    machine_id:
        The machine to display.
    config:
        Loaded dashboard configuration.
    """
    # ── Back button ───────────────────────────────────────────────────────

    if st.button("← Back to Overview"):
        st.session_state["page"] = "Overview"
        st.rerun()

    st.header(f"Equipment Detail — {machine_id}")

    # ── Fetch data ────────────────────────────────────────────────────────

    with st.spinner("Fetching machine data…"):
        machines = fetch_machines(config)
        predictions = fetch_machine_predictions(config, machine_id, hours=24)
        readings = fetch_sensor_readings(config, machine_id, hours=1)
        incidents_resp = fetch_incidents(
            config, machine_id=machine_id, page_size=50
        )

    machine_info = next(
        (m for m in machines if str(m.get("machine_id", "")) == machine_id),
        {},
    )
    incidents = incidents_resp.get("items", [])
    if not isinstance(incidents, list):
        incidents = []

    # ── Machine header ────────────────────────────────────────────────────

    latest_pred = predictions[-1] if predictions else {}
    health_label = str(latest_pred.get("health_label", "unknown"))
    colour = _HEALTH_COLOURS.get(health_label, "#6b7280")
    display_label = health_label.replace("_", " ").title()

    col_id, col_type, col_status, col_date = st.columns(4)
    col_id.metric("Machine ID", machine_id)
    col_type.metric(
        "Type",
        str(machine_info.get("machine_type", "N/A")),
    )
    col_status.markdown(
        f"**Status**\n\n"
        f'<span style="padding:4px 10px;border-radius:4px;'
        f'{colour.replace("#", "background:#")};color:#fff">'
        f"{display_label}</span>",
        unsafe_allow_html=True,
    )
    col_date.metric(
        "Commissioned",
        str(machine_info.get("commissioned_at", "N/A")),
    )

    st.divider()

    # ── RUL trend chart ───────────────────────────────────────────────────

    st.subheader("RUL Trend (Last 24 h)")

    if predictions:
        timestamps: list[datetime] = []
        rul_values: list[float] = []
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []

        for p in predictions:
            dt = _parse_dt(p.get("predicted_at"))
            if dt is None:
                continue
            timestamps.append(dt)
            rul_values.append(
                float(p.get("rul_cycles", 0.0))  # type: ignore[arg-type]
            )
            lower_bounds.append(
                float(p.get("rul_lower_bound", 0.0))  # type: ignore[arg-type]
            )
            upper_bounds.append(
                float(p.get("rul_upper_bound", 0.0))  # type: ignore[arg-type]
            )

        if timestamps:
            fig_rul = rul_trend_chart(
                timestamps,
                rul_values,
                lower_bounds,
                upper_bounds,
                machine_id,
            )
            st.plotly_chart(fig_rul, use_container_width=True)
        else:
            st.info("No timestamped prediction data available.")
    else:
        st.info(
            "No prediction history found for this machine.  "
            "Check that the inference service is running."
        )

    st.divider()

    # ── Health distribution chart ─────────────────────────────────────────

    st.subheader("Health Class Probabilities")

    health_probs: dict[str, float] = {}
    if latest_pred:
        raw_probs = latest_pred.get("health_probabilities", {})
        if isinstance(raw_probs, dict):
            health_probs = {
                str(k): float(v)  # type: ignore[arg-type]
                for k, v in raw_probs.items()
            }

    if health_probs:
        fig_health = health_distribution_chart(health_probs)
        st.plotly_chart(fig_health, use_container_width=True)
    else:
        st.info("Health probability data not yet available for this machine.")

    st.divider()

    # ── Sensor readings grid ──────────────────────────────────────────────

    st.subheader("Sensor Readings (Last Hour)")

    if readings:
        # Collect all sensor names present across readings
        sensor_names: list[str] = []
        for r in readings:
            raw_readings = r.get("readings", {})
            if isinstance(raw_readings, dict):
                for sn in raw_readings:
                    if sn not in sensor_names:
                        sensor_names.append(sn)

        # Prefer key sensors first, then any others
        ordered_sensors = [s for s in _KEY_SENSORS if s in sensor_names] + [
            s for s in sensor_names if s not in _KEY_SENSORS
        ]

        # Show up to 6 sensors in a 3-column grid
        displayed = ordered_sensors[:6]
        if displayed:
            sensor_cols = st.columns(min(3, len(displayed)))
            for idx, sname in enumerate(displayed):
                sensor_ts: list[datetime] = []
                sensor_vals: list[float] = []
                for r in readings:
                    dt = _parse_dt(r.get("recorded_at"))
                    raw_r = r.get("readings", {})
                    if (
                        dt is not None
                        and isinstance(raw_r, dict)
                        and sname in raw_r
                    ):
                        sensor_ts.append(dt)
                        sensor_vals.append(
                            float(raw_r[sname])  # type: ignore[arg-type]
                        )

                col = sensor_cols[idx % 3]
                with col:
                    if sensor_ts:
                        fig_s = sensor_trend_chart(
                            sensor_ts,
                            sensor_vals,
                            sname,
                            machine_id,
                        )
                        st.plotly_chart(
                            fig_s,
                            use_container_width=True,
                        )
    else:
        st.info("No sensor readings available for the last hour.")

    st.divider()

    # ── Incidents table ───────────────────────────────────────────────────

    st.subheader("Incident History")

    if incidents:
        df_inc = pd.DataFrame(incidents)
        display_cols = [
            c
            for c in [
                "incident_id",
                "severity",
                "title",
                "status",
                "triggered_at",
                "resolved_at",
            ]
            if c in df_inc.columns
        ]
        st.dataframe(
            df_inc[display_cols] if display_cols else df_inc,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No incidents found for this machine.")
