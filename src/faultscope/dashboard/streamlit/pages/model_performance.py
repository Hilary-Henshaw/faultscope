"""Model performance and system health page.

Surfaces inference model metadata, prediction distributions, latency
percentiles, Kafka consumer lag, and a link to the MLflow UI.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
import structlog

from faultscope.dashboard.streamlit.components.api_client import (
    fetch_alerting_health,
    fetch_inference_health,
    fetch_inference_models,
    fetch_recent_predictions_sample,
)
from faultscope.dashboard.streamlit.config import DashboardConfig

log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def _rul_histogram(
    rul_values: list[float],
) -> go.Figure:
    """Return a histogram of RUL values from recent predictions."""
    fig = go.Figure(
        go.Histogram(
            x=rul_values,
            nbinsx=20,
            marker_color="#3b82f6",
            opacity=0.8,
            hovertemplate=("RUL bin: %{x}<br>Count: %{y}<extra></extra>"),
        )
    )
    fig.update_layout(
        title="RUL Distribution (Last 100 Predictions)",
        xaxis_title="Remaining Useful Life (cycles)",
        yaxis_title="Count",
        bargap=0.05,
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return fig


def _health_label_bar(
    label_counts: dict[str, int],
) -> go.Figure:
    """Return a bar chart of health-label distribution."""
    ordered = [
        "healthy",
        "degrading",
        "critical",
        "imminent_failure",
    ]
    labels = [lbl for lbl in ordered if lbl in label_counts] + [
        lbl for lbl in label_counts if lbl not in ordered
    ]
    counts = [label_counts.get(lbl, 0) for lbl in labels]
    colours = {
        "healthy": "#22c55e",
        "degrading": "#eab308",
        "critical": "#f97316",
        "imminent_failure": "#ef4444",
    }
    marker_colours = [colours.get(lbl, "#6b7280") for lbl in labels]
    display = [lbl.replace("_", " ").title() for lbl in labels]

    fig = go.Figure(
        go.Bar(
            x=display,
            y=counts,
            marker_color=marker_colours,
            text=counts,
            textposition="outside",
            hovertemplate=("<b>%{x}</b><br>Count: %{y}<extra></extra>"),
        )
    )
    fig.update_layout(
        title="Health Label Distribution (Last 100 Predictions)",
        xaxis_title="Health Label",
        yaxis_title="Count",
        margin={"l": 50, "r": 20, "t": 50, "b": 60},
        showlegend=False,
    )
    return fig


def _latency_gauge(
    label: str,
    value_ms: float,
    max_ms: float = 500.0,
) -> go.Figure:
    """Return a gauge chart for a single latency percentile."""
    colour = (
        "#22c55e"
        if value_ms < max_ms * 0.5
        else "#eab308"
        if value_ms < max_ms * 0.8
        else "#ef4444"
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value_ms,
            title={"text": f"Latency {label} (ms)"},
            gauge={
                "axis": {"range": [0, max_ms]},
                "bar": {"color": colour},
                "steps": [
                    {
                        "range": [0, max_ms * 0.5],
                        "color": "#dcfce7",
                    },
                    {
                        "range": [max_ms * 0.5, max_ms * 0.8],
                        "color": "#fef9c3",
                    },
                    {
                        "range": [max_ms * 0.8, max_ms],
                        "color": "#fee2e2",
                    },
                ],
            },
            number={"suffix": " ms", "valueformat": ".1f"},
        )
    )
    fig.update_layout(
        height=220,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def render_model_performance_page(
    config: DashboardConfig,
) -> None:
    """Render the model performance and system health page.

    Layout
    ------
    1. Current model versions and load status table.
    2. Prediction RUL distribution histogram.
    3. Health-label distribution bar chart.
    4. Inference latency gauges (p50 / p95 / p99).
    5. Kafka consumer lag (from /health endpoint).
    6. Link to MLflow UI.

    Parameters
    ----------
    config:
        Loaded dashboard configuration.
    """
    st.header("Model Performance & System Health")

    # ── Fetch ─────────────────────────────────────────────────────────────

    with st.spinner("Fetching service data…"):
        models = fetch_inference_models(config)
        health_inf = fetch_inference_health(config)
        health_alert = fetch_alerting_health(config)
        recent_preds = fetch_recent_predictions_sample(config, limit=100)

    # ── Model version table ───────────────────────────────────────────────

    st.subheader("Loaded Model Versions")

    if models:
        import pandas as pd

        df_models = pd.DataFrame(models)
        st.dataframe(
            df_models,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(
            "Could not retrieve model information from the "
            "inference API.  Is the service running?"
        )

    st.divider()

    # ── Prediction distribution ───────────────────────────────────────────

    st.subheader("Prediction Distributions")

    if recent_preds:
        rul_values = [
            float(p.get("rul_cycles", 0.0))  # type: ignore[arg-type]
            for p in recent_preds
            if "rul_cycles" in p
        ]
        label_counts: dict[str, int] = {}
        for p in recent_preds:
            lbl = str(p.get("health_label", "unknown"))
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        hist_col, bar_col = st.columns(2)
        if rul_values:
            with hist_col:
                st.plotly_chart(
                    _rul_histogram(rul_values),
                    use_container_width=True,
                )
        if label_counts:
            with bar_col:
                st.plotly_chart(
                    _health_label_bar(label_counts),
                    use_container_width=True,
                )
    else:
        st.info(
            "No recent predictions found.  "
            "Predictions will appear once the inference service "
            "begins processing feature vectors."
        )

    st.divider()

    # ── Latency metrics ───────────────────────────────────────────────────

    st.subheader("Inference Latency")

    latency = health_inf.get("latency_ms", {})
    if not isinstance(latency, dict):
        latency = {}

    p50 = float(latency.get("p50", 0.0))  # type: ignore[arg-type]
    p95 = float(latency.get("p95", 0.0))  # type: ignore[arg-type]
    p99 = float(latency.get("p99", 0.0))  # type: ignore[arg-type]

    lat_col1, lat_col2, lat_col3 = st.columns(3)
    with lat_col1:
        st.plotly_chart(
            _latency_gauge("p50", p50),
            use_container_width=True,
        )
    with lat_col2:
        st.plotly_chart(
            _latency_gauge("p95", p95),
            use_container_width=True,
        )
    with lat_col3:
        st.plotly_chart(
            _latency_gauge("p99", p99),
            use_container_width=True,
        )

    st.divider()

    # ── Kafka consumer lag ────────────────────────────────────────────────

    st.subheader("Kafka Consumer Lag")

    kafka_lag = health_inf.get("kafka_consumer_lag", {})
    if isinstance(kafka_lag, dict) and kafka_lag:
        lag_col1, lag_col2, lag_col3 = st.columns(min(3, len(kafka_lag)))
        lag_cols = [lag_col1, lag_col2, lag_col3]
        for i, (topic, lag) in enumerate(kafka_lag.items()):
            lag_cols[i % 3].metric(
                label=topic,
                value=int(lag),  # type: ignore[arg-type]
                delta=None,
            )
    elif health_inf:
        st.info(
            "Kafka consumer lag data not present in the health "
            "endpoint response."
        )
    else:
        st.warning("Cannot retrieve health data from the inference API.")

    st.divider()

    # ── Alerting service health ───────────────────────────────────────────

    st.subheader("Alerting Service Health")

    alert_status = str(health_alert.get("status", "unknown"))
    if alert_status == "ok":
        st.success(f"Alerting service: {alert_status}")
    elif health_alert:
        st.warning(f"Alerting service status: {alert_status}")
    else:
        st.error("Cannot reach the alerting service health endpoint.")

    st.divider()

    # ── MLflow link ───────────────────────────────────────────────────────

    st.subheader("MLflow Experiment Tracking")
    st.markdown(
        f"Open the MLflow UI: "
        f"[{config.mlflow_tracking_uri}]({config.mlflow_tracking_uri})"
    )
