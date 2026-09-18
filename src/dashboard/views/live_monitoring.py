from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from dashboard.components.theme import (
    BORDER,
    BRAND,
    CRITICAL,
    HIGH,
    LOW,
    MEDIUM,
)
from dashboard.components.ui import render_html

from live_monitoring.inference import (
    ForgeShieldLiveInference,
)
from live_monitoring.simulator import (
    create_machine_fleet,
    generate_fleet_snapshot,
)


# =============================================================
# Helpers
# =============================================================

def _risk_color(band: str) -> str:
    mapping = {
        "Low": LOW,
        "Medium": MEDIUM,
        "High": HIGH,
        "Critical": CRITICAL,
    }

    return mapping.get(
        band,
        "#858C96",
    )


def _risk_counts(
    results: pd.DataFrame,
) -> dict[str, int]:

    counts = {
        "Low": 0,
        "Medium": 0,
        "High": 0,
        "Critical": 0,
    }

    for band, count in (
        results["risk_band"]
        .value_counts()
        .to_dict()
        .items()
    ):
        if band in counts:
            counts[band] = int(count)

    return counts


def _metric_card(
    label: str,
    value: str,
    accent: str,
) -> None:

    render_html(
        f"""
        <div style="
            border:1px solid {BORDER};
            background:#0D1015;
            padding:18px 18px 16px;
            min-height:92px;
        ">
            <div style="
                color:#68707A;
                font-size:9px;
                font-weight:600;
                letter-spacing:0.11em;
                text-transform:uppercase;
            ">
                {label}
            </div>

            <div style="
                color:{accent};
                font-family:'JetBrains Mono',monospace;
                font-size:27px;
                font-weight:600;
                margin-top:9px;
                letter-spacing:-0.04em;
            ">
                {value}
            </div>
        </div>
        """
    )


def _fleet_table(
    results: pd.DataFrame,
) -> None:

    display_columns = [
        "machine_id",
        "Type",
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
        "failure_probability",
        "anomaly_score",
        "unified_risk_score",
        "risk_band",
    ]

    table = results[
        display_columns
    ].copy()

    table.columns = [
        "Machine",
        "Type",
        "Air Temp",
        "Process Temp",
        "RPM",
        "Torque",
        "Tool Wear",
        "Failure Prob.",
        "Anomaly",
        "Unified Risk",
        "Risk Band",
    ]

    table["Air Temp"] = (
        table["Air Temp"].round(2)
    )

    table["Process Temp"] = (
        table["Process Temp"].round(2)
    )

    table["RPM"] = (
        table["RPM"].round(1)
    )

    table["Torque"] = (
        table["Torque"].round(2)
    )

    table["Tool Wear"] = (
        table["Tool Wear"].round(2)
    )

    table["Failure Prob."] = (
        table["Failure Prob."]
        .mul(100)
        .round(2)
        .astype(str)
        + "%"
    )

    table["Anomaly"] = (
        table["Anomaly"]
        .mul(100)
        .round(1)
        .astype(str)
        + "%"
    )

    table["Unified Risk"] = (
        table["Unified Risk"]
        .mul(100)
        .round(1)
        .astype(str)
        + "%"
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Risk Band": st.column_config.TextColumn(
                "Risk Band"
            ),
        },
    )


def _machine_detail(
    results: pd.DataFrame,
    machine_id: str,
) -> None:

    selected = results[
        results["machine_id"] == machine_id
    ]

    if selected.empty:
        return

    row = selected.iloc[0]

    risk = float(
        row["unified_risk_score"]
    )

    band = str(
        row["risk_band"]
    )

    risk_color = _risk_color(
        band
    )

    render_html(
        f"""
        <div style="
            border:1px solid {BORDER};
            background:#0D1015;
            padding:22px;
            margin-top:8px;
        ">

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:flex-start;
                border-bottom:1px solid {BORDER};
                padding-bottom:17px;
                margin-bottom:20px;
            ">

                <div>
                    <div style="
                        color:#F4F5F7;
                        font-size:18px;
                        font-weight:600;
                    ">
                        {machine_id}
                    </div>

                    <div style="
                        color:#68707A;
                        font-size:9px;
                        letter-spacing:0.10em;
                        text-transform:uppercase;
                        margin-top:5px;
                    ">
                        Live Machine Telemetry
                    </div>
                </div>

                <div style="
                    border:1px solid {risk_color};
                    color:{risk_color};
                    padding:6px 12px;
                    font-family:'JetBrains Mono',monospace;
                    font-size:10px;
                    font-weight:600;
                    letter-spacing:0.08em;
                ">
                    {band.upper()}
                </div>

            </div>

            <div style="
                display:grid;
                grid-template-columns:
                    repeat(4, minmax(0, 1fr));
                gap:12px;
            ">

                <div>
                    <div style="color:#68707A;font-size:9px;">
                        AIR TEMPERATURE
                    </div>
                    <div style="
                        color:#E8EAED;
                        font-family:'JetBrains Mono',monospace;
                        font-size:16px;
                        margin-top:5px;
                    ">
                        {float(row["Air temperature"]):.2f} °C
                    </div>
                </div>

                <div>
                    <div style="color:#68707A;font-size:9px;">
                        PROCESS TEMPERATURE
                    </div>
                    <div style="
                        color:#E8EAED;
                        font-family:'JetBrains Mono',monospace;
                        font-size:16px;
                        margin-top:5px;
                    ">
                        {float(row["Process temperature"]):.2f} °C
                    </div>
                </div>

                <div>
                    <div style="color:#68707A;font-size:9px;">
                        ROTATIONAL SPEED
                    </div>
                    <div style="
                        color:#E8EAED;
                        font-family:'JetBrains Mono',monospace;
                        font-size:16px;
                        margin-top:5px;
                    ">
                        {float(row["Rotational speed"]):.1f} RPM
                    </div>
                </div>

                <div>
                    <div style="color:#68707A;font-size:9px;">
                        TORQUE
                    </div>
                    <div style="
                        color:#E8EAED;
                        font-family:'JetBrains Mono',monospace;
                        font-size:16px;
                        margin-top:5px;
                    ">
                        {float(row["Torque"]):.2f} Nm
                    </div>
                </div>

            </div>

            <div style="
                margin-top:25px;
                display:grid;
                grid-template-columns:
                    repeat(3, minmax(0, 1fr));
                gap:12px;
            ">

                <div style="
                    border:1px solid {BORDER};
                    padding:15px;
                ">
                    <div style="
                        color:#68707A;
                        font-size:9px;
                    ">
                        FAILURE PROBABILITY
                    </div>

                    <div style="
                        color:#E8EAED;
                        font-family:'JetBrains Mono',monospace;
                        font-size:22px;
                        margin-top:7px;
                    ">
                        {float(row["failure_probability"]) * 100:.2f}%
                    </div>
                </div>

                <div style="
                    border:1px solid {BORDER};
                    padding:15px;
                ">
                    <div style="
                        color:#68707A;
                        font-size:9px;
                    ">
                        ANOMALY SCORE
                    </div>

                    <div style="
                        color:#E8EAED;
                        font-family:'JetBrains Mono',monospace;
                        font-size:22px;
                        margin-top:7px;
                    ">
                        {float(row["anomaly_score"]) * 100:.1f}%
                    </div>
                </div>

                <div style="
                    border:1px solid {risk_color};
                    padding:15px;
                ">
                    <div style="
                        color:#68707A;
                        font-size:9px;
                    ">
                        UNIFIED RISK
                    </div>

                    <div style="
                        color:{risk_color};
                        font-family:'JetBrains Mono',monospace;
                        font-size:22px;
                        font-weight:600;
                        margin-top:7px;
                    ">
                        {risk * 100:.1f}%
                    </div>
                </div>

            </div>

        </div>
        """
    )


# =============================================================
# Main page
# =============================================================

def render_live_monitoring() -> None:

    # ---------------------------------------------------------
    # Session initialization
    # ---------------------------------------------------------

    if "live_fleet" not in st.session_state:
        st.session_state.live_fleet = (
            create_machine_fleet(
                machine_count=6,
                seed=42,
            )
        )

    if "live_inference" not in st.session_state:
        st.session_state.live_inference = (
            ForgeShieldLiveInference()
        )

    # ---------------------------------------------------------
    # Header
    # ---------------------------------------------------------

    render_html(
        """
        <div style="margin-bottom:5px;">
            <div style="
                color:#F4F5F7;
                font-size:31px;
                font-weight:650;
                letter-spacing:-0.045em;
            ">
                Live Monitoring
            </div>

            <div style="
                color:#858C96;
                font-size:12px;
                margin-top:7px;
            ">
                Real-time multi-machine telemetry and predictive
                risk inference
            </div>
        </div>
        """
    )

    # ---------------------------------------------------------
    # Controls
    # ---------------------------------------------------------

    control_col, status_col = st.columns(
        [1, 3]
    )

    with control_col:

        refresh_seconds = st.selectbox(
            "Refresh interval",
            [1, 2, 3, 5],
            index=1,
            format_func=lambda value:
                f"Every {value} second"
                + ("" if value == 1 else "s"),
        )

    with status_col:

        now = datetime.now().strftime(
            "%H:%M:%S"
        )

        render_html(
            f"""
            <div style="
                height:100%;
                display:flex;
                align-items:center;
                padding:7px 0;
            ">
                <div style="
                    border:1px solid {BORDER};
                    padding:10px 14px;
                    color:#858C96;
                    font-family:'JetBrains Mono',monospace;
                    font-size:10px;
                ">
                    <span style="
                        color:{LOW};
                        font-size:13px;
                    ">●</span>
                    SIMULATED TELEMETRY&nbsp;&nbsp;|&nbsp;&nbsp;
                    LAST UPDATE {now}
                </div>
            </div>
            """
        )

    # ---------------------------------------------------------
    # Generate new telemetry
    # ---------------------------------------------------------

    telemetry = generate_fleet_snapshot(
        st.session_state.live_fleet
    )

    results = (
        st.session_state.live_inference
        .predict(telemetry)
    )

    # ---------------------------------------------------------
    # Fleet summary
    # ---------------------------------------------------------

    counts = _risk_counts(
        results
    )

    average_risk = (
        results["unified_risk_score"]
        .mean()
    )

    st.markdown(
        "<div style='height:16px'></div>",
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)

    with metric_columns[0]:
        _metric_card(
            "Machines Online",
            str(len(results)),
            BRAND,
        )

    with metric_columns[1]:
        _metric_card(
            "Low",
            str(counts["Low"]),
            LOW,
        )

    with metric_columns[2]:
        _metric_card(
            "Medium",
            str(counts["Medium"]),
            MEDIUM,
        )

    with metric_columns[3]:
        _metric_card(
            "High / Critical",
            str(
                counts["High"]
                + counts["Critical"]
            ),
            CRITICAL,
        )

    with metric_columns[4]:
        _metric_card(
            "Average Risk",
            f"{average_risk * 100:.1f}%",
            BRAND,
        )

    # ---------------------------------------------------------
    # Fleet table
    # ---------------------------------------------------------

    st.markdown(
        "<div style='height:28px'></div>",
        unsafe_allow_html=True,
    )

    render_html(
        f"""
        <div style="
            color:#F4F5F7;
            font-size:15px;
            font-weight:600;
            margin-bottom:10px;
        ">
            Machine Fleet
        </div>

        <div style="
            color:#68707A;
            font-size:10px;
            margin-bottom:12px;
        ">
            Continuous telemetry inference across connected
            simulated assets
        </div>
        """
    )

    _fleet_table(
        results
    )

    # ---------------------------------------------------------
    # Selected machine
    # ---------------------------------------------------------

    machine_ids = (
        results["machine_id"]
        .tolist()
    )

    selected_machine = st.selectbox(
        "Inspect machine",
        machine_ids,
        index=0,
    )

    _machine_detail(
        results,
        selected_machine,
    )

    # ---------------------------------------------------------
    # Research note
    # ---------------------------------------------------------

    st.markdown(
        "<div style='height:25px'></div>",
        unsafe_allow_html=True,
    )

    render_html(
        f"""
        <div style="
            border-left:2px solid {BRAND};
            background:#0D1015;
            padding:15px 18px;
        ">
            <div style="
                color:#C4C9D0;
                font-size:11px;
                font-weight:600;
            ">
                POC MONITORING MODE
            </div>

            <div style="
                color:#737B86;
                font-size:10px;
                line-height:1.65;
                margin-top:6px;
            ">
                This demonstration uses simulated machine telemetry
                passed through the same preprocessing and trained
                inference models used by the ForgeShield research
                pipeline. It demonstrates the monitoring and decision
                layer without claiming a production industrial IoT
                deployment.
            </div>
        </div>
        """
    )

    # ---------------------------------------------------------
    # Auto-refresh
    # ---------------------------------------------------------

    time.sleep(
        refresh_seconds
    )

    st.rerun()
