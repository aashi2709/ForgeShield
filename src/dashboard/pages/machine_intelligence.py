from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components.theme import (
    MUTED,
    RISK_COLORS,
)
from dashboard.components.ui import (
    empty_state,
    render_html,
    risk_pill,
)


def _safe_float(value, default=0.0) -> float:
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return default


def _format_sensor(value, decimals=2) -> str:
    try:
        number = float(value)
        if np.isfinite(number):
            return f"{number:,.{decimals}f}"
    except (TypeError, ValueError):
        pass
    return "—"


def render_machine_intelligence(df: pd.DataFrame) -> None:

    render_html(
        """
        <div class="fs-eyebrow">MACHINE INTELLIGENCE</div>
        <div class="fs-page-title">Machine Risk Analysis</div>
        <div class="fs-page-subtitle">
            Inspect failure probability, anomaly evidence, sensor behavior,
            and failure-mode indicators for an individual machine.
        </div>
        """
    )

    st.write("")

    if df.empty:
        render_html(
            empty_state(
                "No machine data",
                "The integrated AI4I risk table contains no observations.",
            )
        )
        return

    # Keep selection deterministic and useful: highest-risk machines first.
    machine_options = (
        df.sort_values(
            "unified_risk_score",
            ascending=False,
        )["Machine ID"]
        .astype(str)
        .tolist()
    )

    selected_machine = st.selectbox(
        "Machine",
        machine_options,
        index=0,
        key="machine_intelligence_selector",
        label_visibility="collapsed",
    )

    selected_rows = df[
        df["Machine ID"].astype(str) == selected_machine
    ]

    if selected_rows.empty:
        render_html(
            empty_state(
                "Machine not found",
                "The selected machine is not available in the current dataset.",
            )
        )
        return

    row = selected_rows.iloc[0]

    risk = str(row.get("risk_band", "Unknown"))
    risk_score = _safe_float(row.get("unified_risk_score"))
    failure_probability = _safe_float(row.get("failure_probability"))
    anomaly_score = _safe_float(row.get("anomaly_score"))
    actual_failure = int(_safe_float(row.get("actual_failure")))

    risk_color = RISK_COLORS.get(risk, MUTED)
    risk_width = max(0.0, min(100.0, risk_score * 100.0))

    render_html(
        f"""
        <div class="fs-machine-hero">
            <div class="fs-machine-hero-top">
                <div>
                    <div class="fs-machine-kicker">SELECTED MACHINE</div>
                    <div class="fs-machine-title">MACHINE {selected_machine}</div>
                    <div class="fs-machine-subtitle">
                        AI4I held-out test observation · Type {row.get("Type", "—")}
                    </div>
                </div>

                <div class="fs-machine-risk">
                    <div class="fs-machine-risk-label">Unified Risk</div>
                    <div class="fs-machine-risk-value" style="color:{risk_color};">
                        {risk_score:.3f}
                    </div>
                    <div class="fs-machine-risk-bar">
                        <div
                            class="fs-machine-risk-fill"
                            style="width:{risk_width:.1f}%;background:{risk_color};"
                        ></div>
                    </div>
                    <div style="margin-top:8px;">
                        {risk_pill(risk)}
                    </div>
                </div>
            </div>
        </div>
        """
    )

    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        render_html(
            f"""
            <div class="fs-detail-card">
                <div class="fs-detail-label">Failure Probability</div>
                <div class="fs-detail-value">{failure_probability:.3f}</div>
                <div class="fs-detail-small">
                    Supervised failure model output
                </div>
            </div>
            """
        )

    with k2:
        render_html(
            f"""
            <div class="fs-detail-card">
                <div class="fs-detail-label">Anomaly Score</div>
                <div class="fs-detail-value">{anomaly_score:.3f}</div>
                <div class="fs-detail-small">
                    Isolation Forest normalized score
                </div>
            </div>
            """
        )

    with k3:
        render_html(
            f"""
            <div class="fs-detail-card">
                <div class="fs-detail-label">Actual Failure</div>
                <div class="fs-detail-value">
                    {"YES" if actual_failure == 1 else "NO"}
                </div>
                <div class="fs-detail-small">
                    Held-out ground-truth label
                </div>
            </div>
            """
        )

    with k4:
        product_type = str(row.get("Type", "—"))

        render_html(
            f"""
            <div class="fs-detail-card">
                <div class="fs-detail-label">Product Type</div>
                <div class="fs-detail-value">{product_type}</div>
                <div class="fs-detail-small">
                    Original AI4I categorical feature
                </div>
            </div>
            """
        )

    render_html(
        """
        <div class="fs-section-title">Sensor Profile</div>
        """
    )

    sensor_fields = [
        ("Air Temperature", "Air temperature", "K"),
        ("Process Temperature", "Process temperature", "K"),
        ("Rotational Speed", "Rotational speed", "rpm"),
        ("Torque", "Torque", "Nm"),
        ("Tool Wear", "Tool wear", "min"),
    ]

    sensor_html = '<div class="fs-sensor-grid">'

    for label, column, unit in sensor_fields:
        value = row.get(column, "—")

        sensor_html += f"""
        <div class="fs-sensor-cell">
            <div class="fs-sensor-name">{label}</div>
            <div class="fs-sensor-value">
                {_format_sensor(value)} {unit}
            </div>
        </div>
        """

    # Engineered features are displayed separately because they are derived,
    # not original sensor measurements.
    sensor_html += f"""
        <div class="fs-sensor-cell">
            <div class="fs-sensor-name">Temperature Differential</div>
            <div class="fs-sensor-value">
                {_format_sensor(row.get("Temperature Differential", 0.0))} K
            </div>
        </div>

        <div class="fs-sensor-cell">
            <div class="fs-sensor-name">Mechanical Power</div>
            <div class="fs-sensor-value">
                {_format_sensor(row.get("Mechanical Power", 0.0))} W
            </div>
        </div>
    """

    sensor_html += "</div>"

    render_html(sensor_html)

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        render_html(
            """
            <div class="fs-section-title">Failure-Mode Indicators</div>
            """
        )

        mode_fields = [
            ("TWF", "Tool Wear Failure"),
            ("HDF", "Heat Dissipation Failure"),
            ("PWF", "Power Failure"),
            ("OSF", "Overstrain Failure"),
            ("RNF", "Random Failure"),
        ]

        mode_html = '<div class="fs-failure-modes">'

        for code, name in mode_fields:
            active = int(_safe_float(row.get(code, 0))) == 1

            mode_html += f"""
            <div class="fs-mode {"fs-mode-active" if active else ""}">
                <span class="fs-mode-name">{name}</span>
                <span class="fs-mode-state">
                    {"FLAGGED" if active else "NOT FLAGGED"}
                </span>
            </div>
            """

        mode_html += "</div>"

        render_html(mode_html)

    with right:
        render_html(
            """
            <div class="fs-section-title">Model Evidence</div>
            """
        )

        if failure_probability >= 0.50 and anomaly_score >= 0.50:
            evidence = (
                "Both model signals are elevated: the supervised model "
                "assigns a high failure probability while the anomaly model "
                "also identifies unusual behavior. The unified score combines "
                "these two signals using the configured 0.60 / 0.40 weighting."
            )
        elif failure_probability >= 0.50:
            evidence = (
                "The supervised model assigns a high failure probability, "
                "while the anomaly score remains below 0.50. This indicates "
                "failure-model evidence without a corresponding strong "
                "unsupervised anomaly signal."
            )
        elif anomaly_score >= 0.50:
            evidence = (
                "The anomaly detector identifies unusual behavior, while "
                "the supervised failure probability remains below 0.50. "
                "This is anomaly evidence rather than a direct failure label."
            )
        else:
            evidence = (
                "Neither the supervised failure probability nor the anomaly "
                "score crosses the 0.50 display threshold for this machine."
            )

        render_html(
            f"""
            <div class="fs-insight">
                {evidence}
            </div>
            """
        )

        st.write("")

        render_html(
            f"""
            <div class="fs-detail-card">
                <div class="fs-detail-label">Risk Composition</div>
                <div style="
                    display:grid;
                    grid-template-columns:1fr 1fr;
                    gap:14px;
                ">
                    <div>
                        <div class="fs-detail-small">Failure · 60%</div>
                        <div class="fs-detail-value" style="font-size:18px;">
                            {0.60 * failure_probability:.3f}
                        </div>
                    </div>

                    <div>
                        <div class="fs-detail-small">Anomaly · 40%</div>
                        <div class="fs-detail-value" style="font-size:18px;">
                            {0.40 * anomaly_score:.3f}
                        </div>
                    </div>
                </div>
            </div>
            """
        )

    render_html(
        """
        <div class="fs-section-title">Machine Ranking</div>
        """
    )

    ranking = (
        df[
            [
                "Machine ID",
                "failure_probability",
                "anomaly_score",
                "unified_risk_score",
                "risk_band",
            ]
        ]
        .sort_values(
            "unified_risk_score",
            ascending=False,
        )
        .head(12)
        .copy()
    )

    ranking_rows = ""

    for rank, (_, machine_row) in enumerate(
        ranking.iterrows(),
        start=1,
    ):
        ranking_rows += f"""
        <tr>
            <td>{rank:02d}</td>
            <td class="machine">{machine_row["Machine ID"]}</td>
            <td class="num">{machine_row["failure_probability"]:.3f}</td>
            <td class="num">{machine_row["anomaly_score"]:.3f}</td>
            <td class="num">{machine_row["unified_risk_score"]:.3f}</td>
            <td class="fs-risk-cell">
                {risk_pill(str(machine_row["risk_band"]))}
            </td>
        </tr>
        """

    render_html(
        f"""
        <div class="fs-panel">
            <div class="fs-panel-header">
                <div class="fs-panel-title">Highest-Risk Machines</div>
                <div class="fs-panel-meta">Top 12 · unified risk</div>
            </div>
            <div class="fs-table-wrap">
                <table class="fs-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Machine ID</th>
                            <th class="num">Failure Probability</th>
                            <th class="num">Anomaly Score</th>
                            <th class="num">Risk Score</th>
                            <th>Risk Band</th>
                        </tr>
                    </thead>
                    <tbody>
                        {ranking_rows}
                    </tbody>
                </table>
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="fs-footer">
            <span>FORGESHIELD · MACHINE INTELLIGENCE</span>
            <span class="fs-footer-mono">
                AI4I held-out test set · v0.1.0
            </span>
        </div>
        """
    )
