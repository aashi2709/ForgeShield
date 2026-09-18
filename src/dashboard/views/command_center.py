from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    render_risk_donut,
    render_scatter,
)
from dashboard.components.ui import (
    build_kpi,
    render_html,
)


def render_command_center(
    df: pd.DataFrame,
    event: dict,
    report: str,
    apply_search_fn: Callable,
    filter_dataset_fn: Callable,
    render_top_three_fn: Callable,
    render_incident_panel_fn: Callable,
    render_alerts_fn: Callable,
    render_activity_table_fn: Callable,
) -> None:

    # Search is deliberately a native input. It remains quiet and functional.
    search_col, refresh_col = st.columns(
        [5.8, 1.25],
        gap="large",
    )

    with search_col:
        search_query = st.text_input(
            "Global search",
            placeholder="Search machines, incidents, or ask a question...",
            key="global_search",
            label_visibility="collapsed",
        )

    with refresh_col:
        now = datetime.now().strftime("%d %b%Y · %H:%M")
        render_html(
            f"""
            <div class="fs-refresh">
                Last refreshed<br>
                <strong>{now}</strong>
            </div>
            """
        )

    st.write("")

    filtered_search_df = apply_search_fn(
        df,
        search_query,
    )

    render_html(
        """
        <div class="fs-eyebrow">COMMAND CENTER</div>
        <div class="fs-page-title">Operational Overview</div>
        <div class="fs-page-subtitle">
            AI-based industrial safety and predictive risk management
        </div>
        """
    )

    st.write("")

    view_col, spacer = st.columns(
        [1.5, 5.5],
    )

    with view_col:
        dataset_view = st.selectbox(
            "Dataset view",
            ["Test Set", "Top 250"],
            index=0,
            key="dataset_view",
        )

    visible_df = filter_dataset_fn(
        filtered_search_df,
        dataset_view,
    )

    st.write("")

    # -----------------------------------------------------------------------
    # KPI row
    # -----------------------------------------------------------------------

    monitored = len(visible_df)

    anomalies = int(
        (visible_df["anomaly_score"] >= 0.50).sum()
    )

    predicted_failures = int(
        (visible_df["failure_probability"] >= 0.50).sum()
    )

    critical_risks = int(
        (visible_df["unified_risk_score"] >= 0.75).sum()
    )

    k1, k2, k3, k4 = st.columns(
        [1, 1, 1, 1],
        gap="large",
    )

    with k1:
        render_html(
            build_kpi(
                "Monitored Machines",
                monitored,
                "0%",
                "AI4I held-out test set",
            )
        )

    with k2:
        render_html(
            build_kpi(
                "Anomalies Detected",
                anomalies,
                "0%",
                "Anomaly score ≥ 0.50",
            )
        )

    with k3:
        render_html(
            build_kpi(
                "Predicted Failures",
                predicted_failures,
                "0%",
                "Failure probability ≥ 0.50",
            )
        )

    with k4:
        render_html(
            build_kpi(
                "Critical Risks",
                critical_risks,
                "0%",
                "Unified risk score ≥ 0.75",
                critical=True,
            )
        )

    st.write("")
    st.write("")

    # -----------------------------------------------------------------------
    # Charts + right rail
    # -----------------------------------------------------------------------

    left, middle, right = st.columns(
        [1.05, 1.05, 0.62],
        gap="large",
    )

    with left:
        render_html(
            f"""
            <div class="fs-panel">
                <div class="fs-panel-header">
                    <div class="fs-panel-title">Risk Distribution</div>
                    <div class="fs-panel-meta">{dataset_view}</div>
                </div>
            </div>
            """
        )

        render_risk_donut(visible_df)

    with middle:
        render_html(
            f"""
            <div class="fs-panel">
                <div class="fs-panel-header">
                    <div class="fs-panel-title">
                        Anomaly Score vs. Failure Probability
                    </div>
                    <div class="fs-panel-meta">{dataset_view}</div>
                </div>
            </div>
            """
        )

        render_scatter(visible_df)

    with right:
        render_html(
            """
            <div class="fs-panel">
                <div class="fs-panel-header">
                    <div class="fs-panel-title">Recent High-Risk Machines</div>
                    <div class="fs-panel-meta">Top 3</div>
                </div>
            """
        )

        render_top_three_fn(visible_df)

        render_html("</div>")

        st.write("")

        render_html(
            """
            <div class="fs-panel fs-incident-panel">
                <div class="fs-panel-header">
                    <div class="fs-panel-title">Latest Incident Report</div>
                    <div class="fs-panel-meta">Current run</div>
                </div>
            """
        )

        render_incident_panel_fn(
            event,
            report,
        )

        render_html("</div>")

        st.write("")

        render_html(
            """
            <div class="fs-panel">
                <div class="fs-panel-header">
                    <div class="fs-panel-title">System Alerts</div>
                    <div class="fs-panel-meta">Current run</div>
                </div>
            """
        )

        render_alerts_fn(
            visible_df,
            event,
        )

        render_html("</div>")

    st.write("")
    st.write("")

    # -----------------------------------------------------------------------
    # Activity table
    # -----------------------------------------------------------------------

    render_html(
        f"""
        <div class="fs-panel">
            <div class="fs-panel-header">
                <div class="fs-panel-title">Recent Machine Activity</div>
                <div class="fs-panel-meta">
                    {dataset_view} · sorted by unified risk
                </div>
            </div>
        </div>
        """
    )

    render_activity_table_fn(
        visible_df,
        limit=14,
    )

    render_html(
        f"""
        <div class="fs-footer">
            <span>FORGESHIELD · INDUSTRIAL SAFETY INTELLIGENCE</span>
            <span class="fs-footer-mono">
                Research Proof of Concept · v0.1.0
            </span>
        </div>
        """
    )
