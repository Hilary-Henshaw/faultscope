"""FaultScope Streamlit multi-page application entry point.

Run with::

    streamlit run src/faultscope/dashboard/streamlit/app.py

Or via the package entry-point::

    python -m faultscope.dashboard
"""

from __future__ import annotations

from datetime import UTC, datetime

import streamlit as st
import structlog

from faultscope.common.logging import configure_logging
from faultscope.dashboard.streamlit.components.api_client import (
    fetch_alerting_health,
    fetch_inference_health,
)
from faultscope.dashboard.streamlit.config import DashboardConfig
from faultscope.dashboard.streamlit.pages.equipment_detail import (
    render_equipment_page,
)
from faultscope.dashboard.streamlit.pages.incidents import (
    render_incidents_page,
)
from faultscope.dashboard.streamlit.pages.model_performance import (
    render_model_performance_page,
)
from faultscope.dashboard.streamlit.pages.overview import (
    render_overview_page,
)

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_PAGES = [
    "Overview",
    "Equipment Detail",
    "Incidents",
    "Model Performance",
]


@st.cache_resource
def _load_config() -> DashboardConfig:
    """Load and cache the dashboard configuration.

    ``st.cache_resource`` ensures this runs exactly once per worker
    process rather than on every rerun.
    """
    configure_logging(level="INFO", fmt="json")
    cfg = DashboardConfig()
    log.info(
        "dashboard_config_loaded",
        inference_url=cfg.inference_base_url,
        alerting_url=cfg.alerting_base_url,
        refresh_s=cfg.refresh_interval_s,
    )
    return cfg


def _status_badge(ok: bool) -> str:
    return "🟢 Online" if ok else "🔴 Offline"


def _sidebar_status(config: DashboardConfig) -> None:
    """Render service-status badges and last-refresh time in sidebar."""
    st.sidebar.subheader("Service Status")

    inf_health = fetch_inference_health(config)
    alt_health = fetch_alerting_health(config)

    inf_ok = bool(inf_health.get("status") == "ok") or bool(inf_health)
    alt_ok = bool(alt_health.get("status") == "ok") or bool(alt_health)

    st.sidebar.markdown(f"**Inference API**: {_status_badge(inf_ok)}")
    st.sidebar.markdown(f"**Alerting API**: {_status_badge(alt_ok)}")

    now = datetime.now(tz=UTC)
    st.sidebar.caption(f"Refreshed: {now.strftime('%H:%M:%S')} UTC")


def main() -> None:
    """Configure the Streamlit page and dispatch to the active page.

    Navigation is stored in ``st.session_state["page"]``.  The sidebar
    ``st.radio`` widget writes back to this key so that any code
    (including machine-card buttons on the overview page) can trigger a
    navigation by setting ``st.session_state["page"]`` and calling
    ``st.rerun()``.
    """
    st.set_page_config(
        page_title="FaultScope — Predictive Maintenance",
        page_icon="🔧",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = _load_config()

    # ── Session state defaults ────────────────────────────────────────────

    if "page" not in st.session_state:
        st.session_state["page"] = "Overview"
    if "selected_machine" not in st.session_state:
        st.session_state["selected_machine"] = ""

    # ── Sidebar ───────────────────────────────────────────────────────────

    with st.sidebar:
        st.image(
            "https://raw.githubusercontent.com/streamlit/streamlit"
            "/develop/frontend/public/favicon.png",
            width=48,
        )
        st.title("FaultScope")
        st.markdown("*Predictive Maintenance Platform*")
        st.divider()

        selected = st.radio(
            "Navigation",
            options=_PAGES,
            index=_PAGES.index(st.session_state.get("page", "Overview")),
            key="nav_radio",
        )
        if selected != st.session_state.get("page"):
            st.session_state["page"] = selected
            st.rerun()

        st.divider()
        _sidebar_status(config)

        st.divider()
        st.caption(
            f"Refresh every {config.refresh_interval_s} s  \n"
            f"Inference: `{config.inference_base_url}`  \n"
            f"Alerting: `{config.alerting_base_url}`"
        )

    # ── Page dispatch ─────────────────────────────────────────────────────

    page = st.session_state.get("page", "Overview")

    try:
        if page == "Overview":
            render_overview_page(config)

        elif page == "Equipment Detail":
            machine_id = st.session_state.get("selected_machine", "")
            if not machine_id:
                st.warning(
                    "No machine selected.  "
                    "Click **Detail** on a machine card from the "
                    "Overview page, or enter a machine ID below."
                )
                machine_id = st.text_input(
                    "Machine ID",
                    placeholder="e.g. FAN-001",
                )
                if machine_id:
                    st.session_state["selected_machine"] = machine_id
                    st.rerun()
            else:
                render_equipment_page(machine_id, config)

        elif page == "Incidents":
            render_incidents_page(config)

        elif page == "Model Performance":
            render_model_performance_page(config)

        else:
            st.error(f"Unknown page: {page!r}")

    except Exception as exc:
        log.exception("dashboard_page_error", page=page, error=str(exc))
        st.error(
            f"An unexpected error occurred while rendering "
            f"**{page}**.  "
            "Check that all FaultScope services are running and "
            "that the API URLs in the configuration are correct.\n\n"
            f"```\n{exc}\n```"
        )


if __name__ == "__main__":
    main()
