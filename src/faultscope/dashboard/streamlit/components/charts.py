"""Reusable Plotly chart factories for the FaultScope dashboard.

Every public function accepts plain Python types and returns a
``plotly.graph_objects.Figure`` ready to be passed to
``st.plotly_chart``.  All figures use a dark-ish, neutral theme that
works well on both light and dark Streamlit themes.
"""

from __future__ import annotations

from datetime import datetime

import plotly.graph_objects as go

# ── colour palette

_HEALTH_COLOURS: dict[str, str] = {
    "healthy": "#22c55e",
    "degrading": "#eab308",
    "critical": "#f97316",
    "imminent_failure": "#ef4444",
}

_CI_FILL_COLOUR = "rgba(59,130,246,0.15)"
_RUL_LINE_COLOUR = "#3b82f6"
_SENSOR_LINE_COLOUR = "#8b5cf6"


def rul_trend_chart(
    timestamps: list[datetime],
    rul_values: list[float],
    lower_bounds: list[float],
    upper_bounds: list[float],
    machine_id: str,
) -> go.Figure:
    """Return a line chart with confidence-interval shading for RUL.

    The shaded band is drawn by combining an upper-bound trace (visible
    line, ``fill=None``) with a lower-bound trace filled *to* the upper
    bound using ``fill='tonexty'``.

    Parameters
    ----------
    timestamps:
        X-axis values (UTC datetimes).
    rul_values:
        Point-estimate RUL in cycles, one per timestamp.
    lower_bounds:
        Lower bound of the prediction interval (cycles).
    upper_bounds:
        Upper bound of the prediction interval (cycles).
    machine_id:
        Used in the chart title.

    Returns
    -------
    go.Figure
        Configured Plotly figure.
    """
    fig = go.Figure()

    # Upper CI boundary (invisible line, anchor for fill)
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=upper_bounds,
            mode="lines",
            line={"width": 0},
            name="CI Upper",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Lower CI boundary filled up to upper
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=lower_bounds,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=_CI_FILL_COLOUR,
            name="95% CI",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # Central RUL line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=rul_values,
            mode="lines+markers",
            line={"color": _RUL_LINE_COLOUR, "width": 2},
            marker={"size": 4},
            name="RUL (cycles)",
            hovertemplate=(
                "<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>"
                "RUL: %{y:.0f} cycles<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"RUL Trend — {machine_id}",
        xaxis_title="Time (UTC)",
        yaxis_title="Remaining Useful Life (cycles)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
    )
    return fig


def health_distribution_chart(
    probabilities: dict[str, float],
) -> go.Figure:
    """Return a horizontal bar chart of health-class probabilities.

    Parameters
    ----------
    probabilities:
        Mapping of health label to softmax probability, e.g.
        ``{"healthy": 0.02, "degrading": 0.10, "critical": 0.85,
        "imminent_failure": 0.03}``.

    Returns
    -------
    go.Figure
        Configured Plotly figure.
    """
    ordered_labels = [
        "healthy",
        "degrading",
        "critical",
        "imminent_failure",
    ]
    labels = [lbl for lbl in ordered_labels if lbl in probabilities] + [
        lbl for lbl in probabilities if lbl not in ordered_labels
    ]
    values = [probabilities.get(lbl, 0.0) for lbl in labels]
    colours = [_HEALTH_COLOURS.get(lbl, "#6b7280") for lbl in labels]
    display_labels = [lbl.replace("_", " ").title() for lbl in labels]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=display_labels,
            orientation="h",
            marker_color=colours,
            text=[f"{v * 100:.1f}%" for v in values],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>Probability: %{x:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Health Class Probabilities",
        xaxis={"title": "Probability", "range": [0, 1.05]},
        yaxis_title="",
        margin={"l": 140, "r": 60, "t": 50, "b": 40},
        showlegend=False,
    )
    return fig


def sensor_trend_chart(
    timestamps: list[datetime],
    values: list[float],
    sensor_name: str,
    machine_id: str,
) -> go.Figure:
    """Return a simple time-series line chart for a single sensor.

    Parameters
    ----------
    timestamps:
        UTC datetimes for the X axis.
    values:
        Sensor measurement values.
    sensor_name:
        Human-readable sensor name shown in the axis label and title.
    machine_id:
        Used in the chart title.

    Returns
    -------
    go.Figure
        Configured Plotly figure.
    """
    fig = go.Figure(
        go.Scatter(
            x=timestamps,
            y=values,
            mode="lines",
            line={"color": _SENSOR_LINE_COLOUR, "width": 1.5},
            name=sensor_name,
            hovertemplate=(
                "<b>%{x|%Y-%m-%d %H:%M:%S}</b><br>"
                f"{sensor_name}: %{{y:.4f}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"{sensor_name} — {machine_id}",
        xaxis_title="Time (UTC)",
        yaxis_title=sensor_name,
        hovermode="x",
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return fig


def equipment_health_heatmap(
    machines: list[dict[str, object]],
) -> go.Figure:
    """Return a grid heatmap coloured by current health status.

    Each cell represents one machine.  The grid is square-ish; the
    number of columns is ``ceil(sqrt(n))``.

    Parameters
    ----------
    machines:
        List of dicts each containing at least:
        ``machine_id`` (str), ``health_label`` (str),
        ``rul_cycles`` (float).

    Returns
    -------
    go.Figure
        Configured Plotly figure.
    """
    import math

    _label_to_z: dict[str, float] = {
        "healthy": 3.0,
        "degrading": 2.0,
        "critical": 1.0,
        "imminent_failure": 0.0,
    }

    n = len(machines)
    if n == 0:
        return go.Figure()

    ncols = max(1, math.ceil(math.sqrt(n)))
    nrows = math.ceil(n / ncols)

    z_grid: list[list[float]] = []
    text_grid: list[list[str]] = []
    customdata_grid: list[list[str]] = []

    for row in range(nrows):
        z_row: list[float] = []
        text_row: list[str] = []
        cd_row: list[str] = []
        for col in range(ncols):
            idx = row * ncols + col
            if idx < n:
                m = machines[idx]
                label = str(m.get("health_label", "healthy"))
                mid = str(m.get("machine_id", ""))
                rul = float(m.get("rul_cycles", 0.0))  # type: ignore[arg-type]
                z_row.append(_label_to_z.get(label, 2.0))
                text_row.append(mid)
                cd_row.append(
                    f"{mid}<br>{label.replace('_', ' ').title()}"
                    f"<br>RUL: {rul:.0f} cycles"
                )
            else:
                z_row.append(float("nan"))
                text_row.append("")
                cd_row.append("")
        z_grid.append(z_row)
        text_grid.append(text_row)
        customdata_grid.append(cd_row)

    colorscale = [
        [0.0, _HEALTH_COLOURS["imminent_failure"]],
        [0.33, _HEALTH_COLOURS["critical"]],
        [0.67, _HEALTH_COLOURS["degrading"]],
        [1.0, _HEALTH_COLOURS["healthy"]],
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z_grid,
            text=text_grid,
            customdata=customdata_grid,
            texttemplate="%{text}",
            colorscale=colorscale,
            zmin=0,
            zmax=3,
            showscale=False,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Equipment Health Grid",
        xaxis={"visible": False},
        yaxis={"visible": False, "autorange": "reversed"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig
