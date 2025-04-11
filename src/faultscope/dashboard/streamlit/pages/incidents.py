"""Incident management page.

Allows operators to browse, filter, acknowledge, and close maintenance
incidents raised by the FaultScope alerting service.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog

from faultscope.dashboard.streamlit.components.api_client import (
    acknowledge_incident,
    close_incident,
    fetch_incidents,
)
from faultscope.dashboard.streamlit.config import DashboardConfig

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_SEVERITY_COLOURS: dict[str, str] = {
    "info": "#3b82f6",
    "warning": "#eab308",
    "critical": "#ef4444",
}

_STATUS_OPTIONS = ["all", "open", "acknowledged", "closed"]
_SEVERITY_OPTIONS = ["all", "info", "warning", "critical"]
_PAGE_SIZE = 50


def _severity_bar_chart(
    df: pd.DataFrame,
) -> go.Figure:
    """Return a bar chart showing incident count by severity."""
    counts = (
        df["severity"]
        .value_counts()
        .reindex(["info", "warning", "critical"], fill_value=0)
        if "severity" in df.columns
        else pd.Series({"info": 0, "warning": 0, "critical": 0})
    )
    colours = [_SEVERITY_COLOURS.get(s, "#6b7280") for s in counts.index]
    fig = go.Figure(
        go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker_color=colours,
            text=counts.values.tolist(),
            textposition="outside",
            hovertemplate=("<b>%{x}</b><br>Count: %{y}<extra></extra>"),
        )
    )
    fig.update_layout(
        title="Incident Count by Severity",
        xaxis_title="Severity",
        yaxis_title="Count",
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        showlegend=False,
    )
    return fig


def render_incidents_page(config: DashboardConfig) -> None:
    """Render the alert/incident management page.

    Layout
    ------
    1. Filter controls: machine_id, status, severity.
    2. Severity distribution bar chart.
    3. Paginated incidents table with Acknowledge / Close buttons.

    Parameters
    ----------
    config:
        Loaded dashboard configuration.
    """
    st.header("Incident Management")

    # ── Filters ───────────────────────────────────────────────────────────

    with st.expander("Filters", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        machine_filter = f_col1.text_input(
            "Machine ID", value="", placeholder="e.g. FAN-001"
        )
        status_filter = f_col2.selectbox(
            "Status", options=_STATUS_OPTIONS, index=0
        )
        severity_filter = f_col3.selectbox(
            "Severity", options=_SEVERITY_OPTIONS, index=0
        )

    # Resolve "all" → None for API
    status_param = None if status_filter == "all" else status_filter
    severity_param = None if severity_filter == "all" else severity_filter
    machine_param = machine_filter.strip() or None

    # Pagination state
    if "incidents_page" not in st.session_state:
        st.session_state["incidents_page"] = 1
    page: int = st.session_state["incidents_page"]

    # ── Fetch ─────────────────────────────────────────────────────────────

    with st.spinner("Loading incidents…"):
        result = fetch_incidents(
            config,
            machine_id=machine_param,
            status=status_param,
            severity=severity_param,
            page=page,
            page_size=_PAGE_SIZE,
        )

    items = result.get("items", [])
    if not isinstance(items, list):
        items = []
    total = int(result.get("total", 0))  # type: ignore[arg-type, call-overload]
    total_pages = int(result.get("pages", 1))  # type: ignore[arg-type, call-overload]

    # ── Summary chart ─────────────────────────────────────────────────────

    if items:
        df_all = pd.DataFrame(items)
        fig = _severity_bar_chart(df_all)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Showing page {page} of {max(total_pages, 1)} "
        f"({total} total incidents)"
    )

    # ── Incidents table with action buttons ───────────────────────────────

    if not items:
        st.info("No incidents match the selected filters.")
        return

    df = pd.DataFrame(items)
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
        if c in df.columns
    ]

    # Show the read-only table
    st.dataframe(
        df[display_cols] if display_cols else df,
        use_container_width=True,
        hide_index=True,
    )

    # Action buttons per incident
    st.subheader("Actions")
    for inc in items:
        inc_id = str(inc.get("incident_id", ""))
        inc_status = str(inc.get("status", ""))
        inc_title = str(inc.get("title", inc_id))
        if not inc_id:
            continue

        with st.expander(
            f"{inc.get('severity', '').upper()} — {inc_title}",
            expanded=False,
        ):
            a_col, c_col, _ = st.columns([1, 1, 4])
            if inc_status == "open":
                if a_col.button(
                    "Acknowledge",
                    key=f"ack_{inc_id}",
                    type="primary",
                ):
                    if acknowledge_incident(config, inc_id):
                        st.success(f"Incident {inc_id} acknowledged.")
                        log.info(
                            "incident_acknowledged",
                            incident_id=inc_id,
                        )
                    else:
                        st.error(
                            "Failed to acknowledge.  "
                            "Check alerting API connectivity."
                        )
            if inc_status in ("open", "acknowledged"):
                if c_col.button("Close", key=f"close_{inc_id}"):
                    if close_incident(config, inc_id):
                        st.success(f"Incident {inc_id} closed.")
                        log.info(
                            "incident_closed",
                            incident_id=inc_id,
                        )
                    else:
                        st.error(
                            "Failed to close.  "
                            "Check alerting API connectivity."
                        )

    # ── Pagination controls ───────────────────────────────────────────────

    st.divider()
    nav_prev, nav_page, nav_next = st.columns([1, 2, 1])
    if nav_prev.button(
        "◀ Previous",
        disabled=(page <= 1),
        key="inc_prev",
    ):
        st.session_state["incidents_page"] = max(1, page - 1)
        st.rerun()
    nav_page.markdown(
        f"<div style='text-align:center'>Page {page} / "
        f"{max(total_pages, 1)}</div>",
        unsafe_allow_html=True,
    )
    if nav_next.button(
        "Next ▶",
        disabled=(page >= total_pages),
        key="inc_next",
    ):
        st.session_state["incidents_page"] = page + 1
        st.rerun()
