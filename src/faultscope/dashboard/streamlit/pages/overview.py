"""Overview page: equipment health grid, summary metrics, incidents.

This is the default landing page of the FaultScope dashboard.  It
shows a high-level view of all machines in the fleet and any currently
open incidents.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st
import structlog

from faultscope.dashboard.streamlit.components.api_client import (
    fetch_active_incidents,
    fetch_inference_models,
    fetch_latest_predictions,
    fetch_machines,
)
from faultscope.dashboard.streamlit.components.charts import (
    equipment_health_heatmap,
)
from faultscope.dashboard.streamlit.config import DashboardConfig

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ── Colour mapping for health labels ─────────────────────────────────────────

_HEALTH_BADGE: dict[str, str] = {
    "healthy": "🟢",
    "degrading": "🟡",
    "critical": "🟠",
    "imminent_failure": "🔴",
}

_HEALTH_CSS: dict[str, str] = {
    "healthy": "background:#22c55e;color:#fff",
    "degrading": "background:#eab308;color:#000",
    "critical": "background:#f97316;color:#fff",
    "imminent_failure": "background:#ef4444;color:#fff",
}


def _label_colour(label: str) -> str:
    return _HEALTH_CSS.get(label, "background:#6b7280;color:#fff")


def _badge(label: str) -> str:
    return _HEALTH_BADGE.get(label, "⚪")


def render_overview_page(config: DashboardConfig) -> None:
    """Render the equipment health overview page.

    Layout
    ------
    1. Page header with last-updated timestamp.
    2. Four summary metric cards: Total / Critical / Warning / Healthy.
    3. Equipment health grid heatmap.
    4. Individual machine cards in a responsive column grid.
    5. Active incidents table.
    6. Auto-refresh via ``st.rerun`` + ``time.sleep``.

    Parameters
    ----------
    config:
        Loaded dashboard configuration.
    """
    st.header("Equipment Health Overview")

    now_utc = datetime.now(tz=UTC)
    st.caption(f"Last updated: {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # ── Fetch data ────────────────────────────────────────────────────────

    with st.spinner("Fetching machine data…"):
        machines = fetch_machines(config)
        predictions = fetch_latest_predictions(config)
        incidents = fetch_active_incidents(config)

    # Build a prediction lookup: machine_id → latest prediction dict
    pred_by_machine: dict[str, dict[str, object]] = {}
    for p in predictions:
        mid = str(p.get("machine_id", ""))
        if mid:
            pred_by_machine[mid] = p

    # Merge machine info with predictions
    enriched: list[dict[str, object]] = []
    for m in machines:
        mid = str(m.get("machine_id", ""))
        pred = pred_by_machine.get(mid, {})
        enriched.append(
            {
                "machine_id": mid,
                "machine_type": m.get("machine_type", "unknown"),
                "health_label": pred.get("health_label", "healthy"),
                "rul_cycles": float(
                    pred.get("rul_cycles", 0.0)  # type: ignore[arg-type]
                ),
                "rul_hours": float(
                    pred.get("rul_hours", 0.0)  # type: ignore[arg-type]
                ),
                "anomaly_score": float(
                    pred.get("anomaly_score", 0.0)  # type: ignore[arg-type]
                ),
                "predicted_at": pred.get("predicted_at", ""),
            }
        )

    # ── Summary metric cards ──────────────────────────────────────────────

    total = len(enriched)
    critical_count = sum(
        1
        for m in enriched
        if m["health_label"] in ("critical", "imminent_failure")
    )
    warning_count = sum(
        1 for m in enriched if m["health_label"] == "degrading"
    )
    healthy_count = sum(1 for m in enriched if m["health_label"] == "healthy")

    col_total, col_critical, col_warning, col_healthy = st.columns(4)
    col_total.metric("Total Machines", total)
    col_critical.metric(
        "Critical / Imminent",
        critical_count,
        delta=(f"+{critical_count}" if critical_count > 0 else None),
        delta_color="inverse",
    )
    col_warning.metric(
        "Degrading",
        warning_count,
        delta=(f"+{warning_count}" if warning_count > 0 else None),
        delta_color="inverse",
    )
    col_healthy.metric("Healthy", healthy_count)

    st.divider()

    # ── Heatmap overview ──────────────────────────────────────────────────

    if enriched:
        fig = equipment_health_heatmap(enriched)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No machine data available.  "
            "Verify that the inference API is reachable."
        )

    st.divider()

    # ── Machine cards grid ────────────────────────────────────────────────

    st.subheader("Machine Status Cards")

    cards_per_row = 4
    if enriched:
        rows = [
            enriched[i : i + cards_per_row]
            for i in range(0, len(enriched), cards_per_row)
        ]
        for row_machines in rows:
            cols = st.columns(len(row_machines))
            for col, m in zip(cols, row_machines, strict=False):
                label = str(m["health_label"])
                badge = _badge(label)
                rul_c = float(m["rul_cycles"])  # type: ignore[arg-type]
                rul_h = float(m["rul_hours"])  # type: ignore[arg-type]
                display_label = label.replace("_", " ").title()
                with col:
                    st.markdown(
                        f"**{m['machine_id']}**  "
                        f"{badge} {display_label}\n\n"
                        f"RUL: **{rul_c:.0f}** cycles "
                        f"/ **{rul_h:.1f}** h"
                    )
                    if st.button(
                        "Detail",
                        key=f"detail_{m['machine_id']}",
                    ):
                        st.session_state["selected_machine"] = m["machine_id"]
                        st.session_state["page"] = "Equipment Detail"
                        st.rerun()
    else:
        st.info("No machines registered yet.")

    st.divider()

    # ── Active incidents table ────────────────────────────────────────────

    st.subheader("Active Incidents")

    if incidents:
        df_inc = pd.DataFrame(incidents)
        display_cols = [
            c
            for c in [
                "incident_id",
                "machine_id",
                "severity",
                "title",
                "status",
                "triggered_at",
            ]
            if c in df_inc.columns
        ]
        st.dataframe(
            df_inc[display_cols] if display_cols else df_inc,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No active incidents.")

    # ── Sidebar: model status ─────────────────────────────────────────────

    with st.sidebar:
        st.subheader("Model Status")
        models = fetch_inference_models(config)
        if models:
            for model in models:
                name = str(model.get("name", "unknown"))
                version = str(model.get("version", "?"))
                status = str(model.get("status", "unknown"))
                badge = "✅" if status == "loaded" else "⚠️"
                st.markdown(f"{badge} **{name}** v{version}")
        else:
            st.warning("Cannot reach inference API.")

    # ── Auto-refresh ──────────────────────────────────────────────────────

    import time

    time.sleep(config.refresh_interval_s)
    st.rerun()
