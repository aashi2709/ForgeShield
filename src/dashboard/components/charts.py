from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.theme import (
    BACKGROUND,
    BORDER,
    BORDER_STRONG,
    RISK_COLORS,
    SURFACE,
    TEXT,
)
from dashboard.components.ui import (
    empty_state,
    render_html,
)


def render_risk_donut(df: pd.DataFrame) -> None:

    counts = (
        df["risk_band"]
        .value_counts()
        .reindex(
            ["Low", "Medium", "High", "Critical"],
            fill_value=0,
        )
    )

    total = int(counts.sum())

    if total == 0:
        render_html(
            empty_state(
                "No risk observations",
                "There is no data available for this view.",
            )
        )
        return

    labels = counts.index.tolist()
    values = counts.values.tolist()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.67,
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=[
                        RISK_COLORS[label]
                        for label in labels
                    ],
                    line=dict(
                        color=BACKGROUND,
                        width=2,
                    ),
                ),
                textinfo="percent",
                textposition="inside",
                textfont=dict(
                    color=BACKGROUND,
                    size=11,
                ),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "%{value:,} machines<br>"
                    "%{percent}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        height=310,
        margin=dict(
            l=12,
            r=12,
            t=10,
            b=8,
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        showlegend=False,
        annotations=[
            dict(
                text=(
                    f"<b>{total:,}</b>"
                    "<br><span style='font-size:10px'>machines</span>"
                ),
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(
                    family="JetBrains Mono",
                    size=23,
                    color=TEXT,
                ),
            )
        ],
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )

    legend_html = '<div class="fs-legend">'

    for label in labels:
        legend_html += f"""
        <div class="fs-legend-item">
            <div class="fs-legend-name">{label}</div>
            <div class="fs-legend-count">{counts[label]:,}</div>
        </div>
        """

    legend_html += "</div>"

    render_html(legend_html)


def render_scatter(df: pd.DataFrame) -> None:

    if df.empty:
        render_html(
            empty_state(
                "No observations",
                "The selected view contains no machines.",
            )
        )
        return

    plot_df = df.copy()

    fig = go.Figure()

    for risk in ["Low", "Medium", "High", "Critical"]:

        subset = plot_df[
            plot_df["risk_band"] == risk
        ]

        if subset.empty:
            continue

        hover_machine = subset["Machine ID"].astype(str)

        fig.add_trace(
            go.Scatter(
                x=subset["failure_probability"],
                y=subset["anomaly_score"],
                mode="markers",
                name=risk,
                text=hover_machine,
                customdata=np.stack(
                    [
                        subset["unified_risk_score"],
                        subset["failure_probability"],
                        subset["anomaly_score"],
                    ],
                    axis=1,
                ),
                marker=dict(
                    size=6,
                    color=RISK_COLORS[risk],
                    opacity=0.86,
                    line=dict(
                        width=0,
                    ),
                ),
                hovertemplate=(
                    "<b>Machine %{text}</b><br>"
                    "Failure probability: %{customdata[1]:.3f}<br>"
                    "Anomaly score: %{customdata[2]:.3f}<br>"
                    "Unified risk: %{customdata[0]:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    # Critical zone: flat low-opacity fill, no gradient or glow.
    fig.add_shape(
        type="rect",
        x0=0.50,
        x1=1.00,
        y0=0.50,
        y1=1.00,
        line=dict(
            width=0,
        ),
        fillcolor="rgba(239,75,92,0.075)",
        layer="below",
    )

    fig.add_shape(
        type="line",
        x0=0.50,
        x1=0.50,
        y0=0,
        y1=1,
        line=dict(
            color="#4D535D",
            width=1,
            dash="dash",
        ),
    )

    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0.50,
        y1=0.50,
        line=dict(
            color="#4D535D",
            width=1,
            dash="dash",
        ),
    )

    fig.add_annotation(
        x=0.94,
        y=0.95,
        text="CRITICAL ZONE",
        showarrow=False,
        font=dict(
            family="Inter",
            size=9,
            color="#AEB4BC",
        ),
        xanchor="right",
    )

    fig.update_layout(
        height=350,
        margin=dict(
            l=48,
            r=14,
            t=16,
            b=45,
        ),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        hoverlabel=dict(
            bgcolor="#181D24",
            bordercolor=BORDER_STRONG,
            font=dict(
                family="Inter",
                color=TEXT,
                size=11,
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(
                family="Inter",
                size=9,
                color="#8F969F",
            ),
        ),
        xaxis=dict(
            title=dict(
                text="Failure Probability",
                font=dict(
                    size=11,
                    color="#A0A6AE",
                ),
            ),
            range=[0, 1],
            tickfont=dict(
                size=9,
                color="#737B85",
            ),
            gridcolor="rgba(70,76,85,0.18)",
            zeroline=False,
            linecolor=BORDER,
        ),
        yaxis=dict(
            title=dict(
                text="Anomaly Score",
                font=dict(
                    size=11,
                    color="#A0A6AE",
                ),
            ),
            range=[0, 1],
            tickfont=dict(
                size=9,
                color="#737B85",
            ),
            gridcolor="rgba(70,76,85,0.18)",
            zeroline=False,
            linecolor=BORDER,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )
