from pathlib import Path
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# ForgeShield dashboard bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from risk_scoring import calculate_risk_band  # noqa: E402
from dashboard.components.theme import load_global_css  # noqa: E402
from dashboard.components.data import (
    load_risk_table as _load_risk_table,
    load_test_metadata as _load_test_metadata,
    load_demo_event as _load_demo_event,
    load_incident_report as _load_incident_report,
    load_validation as _load_validation,
)  # noqa: E402

from dashboard.components.ui import (
    render_html,
    risk_class,
    risk_pill,
    metric_number,
    build_kpi,
    empty_state,
)  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INTEGRATION_DIR = PROJECT_ROOT / "reports" / "integration"

RISK_TABLE_PATH = INTEGRATION_DIR / "integrated_risk_table.csv"
EVENT_PATH = INTEGRATION_DIR / "demo_event.json"
REPORT_PATH = INTEGRATION_DIR / "incident_report.md"
VALIDATION_PATH = INTEGRATION_DIR / "incident_report_validation.json"

TEST_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ai4i"
    / "test_metadata.csv"
)


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

BG = "#0A0C10"
SURFACE = "#12151A"
SURFACE_2 = "#151920"
BORDER = "#1C2028"
BORDER_STRONG = "#292F38"

TEXT = "#E7E9EC"
MUTED = "#868D97"
MUTED_2 = "#646B75"

BRAND = "#FF6B35"

CRITICAL = "#EF4B5C"
HIGH = "#F0A63C"
MEDIUM = "#E8C547"
LOW = "#38C98F"

RISK_COLORS = {
    "Low": LOW,
    "Medium": MEDIUM,
    "High": HIGH,
    "Critical": CRITICAL,
}

PAGE_RADIUS = "8px"
PANEL_RADIUS = "10px"
SMALL_RADIUS = "4px"


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ForgeShield Command Center",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# HTML helper
#
# IMPORTANT:
# Use st.html(), not st.markdown(..., unsafe_allow_html=True), for custom HTML.
# Indented HTML passed through Markdown can be interpreted as a code block.
# ---------------------------------------------------------------------------

def render_html(html: str) -> None:
    st.html(html.strip())


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

GLOBAL_CSS = load_global_css()
render_html(GLOBAL_CSS)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_risk_table() -> pd.DataFrame:
    return _load_risk_table(RISK_TABLE_PATH)


@st.cache_data(show_spinner=False)
def load_test_metadata() -> pd.DataFrame:
    return _load_test_metadata(TEST_METADATA_PATH)


@st.cache_data(show_spinner=False)
def load_demo_event() -> dict:
    return _load_demo_event(EVENT_PATH)


@st.cache_data(show_spinner=False)
def load_incident_report() -> str:
    return _load_incident_report(REPORT_PATH)


@st.cache_data(show_spinner=False)
def load_validation() -> dict:
    return _load_validation(VALIDATION_PATH)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(
    risk_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:

    df = risk_df.copy()

    if not metadata_df.empty:
        metadata = metadata_df.copy()

        if len(metadata) == len(df):
            preferred = [
                "UID",
                "Type",
                "Air temperature",
                "Process temperature",
                "Rotational speed",
                "Torque",
                "Tool wear",
                "Machine failure",
                "TWF",
                "HDF",
                "PWF",
                "OSF",
                "RNF",
            ]

            available = [
                column
                for column in preferred
                if column in metadata.columns
            ]

            metadata = metadata[available].copy()

            # The risk table can already contain a few raw columns such as
            # UID or Type. Remove overlapping columns before concatenating so
            # pandas does not create duplicate column names. Duplicate names
            # would make df["Air temperature"] return a DataFrame instead of
            # a Series and cause pd.to_numeric() to fail.
            overlapping = [
                column
                for column in metadata.columns
                if column in df.columns
            ]
            if overlapping:
                df = df.drop(columns=overlapping)

            df = pd.concat(
                [
                    metadata.reset_index(drop=True),
                    df.reset_index(drop=True),
                ],
                axis=1,
            )

            # Reconstruct engineered features from the original AI4I
            # measurements for the Machine Intelligence view. The risk table
            # contains model outputs, while test_metadata contains the raw
            # held-out observation.
            if (
                "Air temperature" in df.columns
                and "Process temperature" in df.columns
            ):
                air_temperature = pd.to_numeric(
                    df["Air temperature"], errors="coerce"
                )
                process_temperature = pd.to_numeric(
                    df["Process temperature"], errors="coerce"
                )
                df["Temperature Differential"] = (
                    process_temperature - air_temperature
                )

            if (
                "Rotational speed" in df.columns
                and "Torque" in df.columns
            ):
                rotational_speed = pd.to_numeric(
                    df["Rotational speed"], errors="coerce"
                )
                torque = pd.to_numeric(
                    df["Torque"], errors="coerce"
                )
                df["Mechanical Power"] = (
                    rotational_speed * torque * (2.0 * np.pi / 60.0)
                )

    if "UID" not in df.columns:
        df.insert(
            0,
            "UID",
            np.arange(1, len(df) + 1),
        )

    df["Machine ID"] = (
        df["UID"]
        .astype(str)
        .str.replace(".0", "", regex=False)
    )

    df["failure_probability"] = pd.to_numeric(
        df["failure_probability"],
        errors="coerce",
    )

    df["anomaly_score"] = pd.to_numeric(
        df["anomaly_score"],
        errors="coerce",
    )

    df["unified_risk_score"] = pd.to_numeric(
        df["unified_risk_score"],
        errors="coerce",
    )

    df["actual_failure"] = pd.to_numeric(
        df["actual_failure"],
        errors="coerce",
    ).fillna(0).astype(int)

    return df


def filter_dataset(
    df: pd.DataFrame,
    dataset_view: str,
) -> pd.DataFrame:

    if dataset_view == "Top 250":
        return (
            df.sort_values(
                "unified_risk_score",
                ascending=False,
            )
            .head(250)
            .copy()
        )

    return df.copy()


def apply_search(
    df: pd.DataFrame,
    query: str,
) -> pd.DataFrame:

    query = query.strip()

    if not query:
        return df

    mask = (
        df["Machine ID"]
        .astype(str)
        .str.contains(
            query,
            case=False,
            na=False,
        )
    )

    if "Type" in df.columns:
        mask = mask | (
            df["Type"]
            .astype(str)
            .str.contains(
                query,
                case=False,
                na=False,
            )
        )

    if "risk_band" in df.columns:
        mask = mask | (
            df["risk_band"]
            .astype(str)
            .str.contains(
                query,
                case=False,
                na=False,
            )
        )

    return df[mask].copy()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

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
                        color=BG,
                        width=2,
                    ),
                ),
                textinfo="percent",
                textposition="inside",
                textfont=dict(
                    color=BG,
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


# ---------------------------------------------------------------------------
# Tables / rail
# ---------------------------------------------------------------------------

def render_activity_table(df: pd.DataFrame, limit: int = 14) -> None:

    if df.empty:
        render_html(
            empty_state(
                "No matching machines",
                "Try clearing the search or selecting a different dataset view.",
            )
        )
        return

    table_df = (
        df.sort_values(
            "unified_risk_score",
            ascending=False,
        )
        .head(limit)
        .copy()
    )

    rows = ""

    for _, row in table_df.iterrows():

        risk = str(row["risk_band"])

        rows += f"""
        <tr>
            <td class="machine">{row["Machine ID"]}</td>
            <td class="num">{row["failure_probability"]:.3f}</td>
            <td class="num">{row["anomaly_score"]:.3f}</td>
            <td class="num">{row["unified_risk_score"]:.3f}</td>
            <td class="fs-risk-cell">{risk_pill(risk)}</td>
        </tr>
        """

    html = f"""
    <div class="fs-table-wrap">
        <table class="fs-table">
            <thead>
                <tr>
                    <th>Machine ID <span class="fs-sort">↕</span></th>
                    <th class="num">Failure Probability <span class="fs-sort">↕</span></th>
                    <th class="num">Anomaly Score <span class="fs-sort">↕</span></th>
                    <th class="num">Risk Score <span class="fs-sort">↕</span></th>
                    <th>Risk Band <span class="fs-sort">↕</span></th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
    """

    render_html(html)


def render_top_three(df: pd.DataFrame) -> None:

    top = (
        df.sort_values(
            "unified_risk_score",
            ascending=False,
        )
        .head(3)
    )

    if top.empty:
        render_html(
            empty_state(
                "No high-risk machines",
                "No machines are available in this view.",
            )
        )
        return

    html = ""

    for _, row in top.iterrows():
        risk = str(row["risk_band"])

        html += f"""
        <div class="fs-rail-item">
            <span class="fs-machine-id">
                {row["Machine ID"]}
            </span>
            {risk_pill(risk)}
        </div>
        """

    render_html(html)


def render_incident_panel(event: dict, report: str) -> None:

    if not event:
        render_html(
            empty_state(
                "No incident event",
                "Run the integration pipeline to generate a demo event.",
            )
        )
        return

    machine_id = event.get("machine_id", "—")
    event_type = event.get("event_type", "—")
    risk_band = event.get("risk_band", "—")
    risk_score = float(event.get("risk_score", 0.0))

    summary = (
        "Integrated event generated from the AI4I held-out test set. "
        "Incident intelligence is grounded using retrieved historical "
        "incidents and safety procedures."
    )

    if report:
        first_lines = [
            line.strip()
            for line in report.splitlines()
            if line.strip()
        ]

        if first_lines:
            summary = (
                " ".join(first_lines[:2])
                .replace("**", "")
                .strip()
            )[:250]

    html = f"""
    <div class="fs-incident-grid">
        <div>
            <div class="fs-incident-label">Machine ID</div>
            <div class="fs-incident-value">{machine_id}</div>
        </div>

        <div>
            <div class="fs-incident-label">Event Type</div>
            <div class="fs-incident-value">{event_type}</div>
        </div>

        <div>
            <div class="fs-incident-label">Risk Band</div>
            <div>{risk_pill(risk_band)}</div>
        </div>

        <div>
            <div class="fs-incident-label">Risk Score</div>
            <div class="fs-incident-value">{risk_score:.3f}</div>
        </div>
    </div>

    <div class="fs-incident-summary">
        {summary}
    </div>
    """

    render_html(html)


def render_alerts(
    df: pd.DataFrame,
    event: dict,
) -> None:

    critical_count = int(
        (df["risk_band"] == "Critical").sum()
    )

    high_count = int(
        (df["risk_band"] == "High").sum()
    )

    anomaly_count = int(
        (df["anomaly_score"] >= 0.50).sum()
    )

    machine_id = (
        str(event.get("machine_id", "—"))
        if event
        else "—"
    )

    alerts = [
        (
            CRITICAL,
            f"High risk detected · Machine {machine_id}",
        ),
        (
            HIGH,
            f"{high_count:,} machines currently classified High risk",
        ),
        (
            MEDIUM,
            f"{anomaly_count:,} machines exceed anomaly threshold",
        ),
        (
            LOW,
            "AI4I + C-MAPSS data sources available",
        ),
    ]

    html = ""

    for color, message in alerts:
        html += f"""
        <div class="fs-alert">
            <div
                class="fs-alert-bar"
                style="background:{color};"
            ></div>
            <div class="fs-alert-text">
                {message}
            </div>
        </div>
        """

    render_html(html)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> str:
    """Render all navigation chrome inside the actual Streamlit sidebar.

    Keeping the entire block inside ``with st.sidebar`` is important: custom
    HTML must never fall back into the main Command Center column.
    """

    navigation = [
        "Command Center",
        "Machine Intelligence",
        "Predictive Health",
        "Anomaly Detection",
        "Explainability",
        "Incident Intelligence",
        "Safety Copilot",
        "Knowledge Base",
        "Documentation",
        "Settings",
    ]

    with st.sidebar:
        render_html(
            f"""
            <div style="
                padding: 28px 0 25px;
                border-bottom: 1px solid {BORDER};
                margin-bottom: 23px;
            ">
                <div style="
                    display:flex;
                    align-items:center;
                    gap:10px;
                ">
                    <span style="
                        display:inline-flex;
                        width:15px;
                        height:15px;
                        border:2px solid {BRAND};
                        transform:rotate(45deg);
                        box-sizing:border-box;
                    "></span>

                    <span style="
                        color:#F4F5F7;
                        font-size:19px;
                        font-weight:700;
                        letter-spacing:-0.04em;
                    ">
                        ForgeShield
                    </span>
                </div>

                <div style="
                    color:#858C96;
                    font-size:9px;
                    font-weight:500;
                    letter-spacing:0.05em;
                    margin:8px 0 0 25px;
                ">
                    INDUSTRIAL SAFETY INTELLIGENCE
                </div>
            </div>

            <div style="
                color:#68707A;
                font-size:9px;
                font-weight:600;
                letter-spacing:0.11em;
                text-transform:uppercase;
                margin-bottom:9px;
            ">
                Navigation
            </div>
            """
        )

        selected = st.radio(
            "Navigation",
            navigation,
            index=0,
            key="forge_navigation",
            label_visibility="collapsed",
        )

        render_html(
            f"""
            <div style="
                margin-top:30px;
                padding-top:18px;
                border-top:1px solid {BORDER};
            ">
                <div style="
                    color:#68707A;
                    font-size:9px;
                    font-weight:600;
                    letter-spacing:0.11em;
                    text-transform:uppercase;
                    margin-bottom:12px;
                ">
                    System Status
                </div>

                <div style="display:grid;gap:12px;">
                    <div>
                        <div style="display:flex;align-items:center;gap:8px;color:#8B929C;font-size:11px;">
                            <span style="width:6px;height:6px;border-radius:50%;background:{LOW};"></span>
                            AI4I + C-MAPSS
                        </div>
                        <div style="color:{LOW};font-family:'JetBrains Mono',monospace;font-size:9px;margin:3px 0 0 14px;">Online</div>
                    </div>

                    <div>
                        <div style="display:flex;align-items:center;gap:8px;color:#8B929C;font-size:11px;">
                            <span style="width:6px;height:6px;border-radius:50%;background:{LOW};"></span>
                            RAG Knowledge Base
                        </div>
                        <div style="color:{LOW};font-family:'JetBrains Mono',monospace;font-size:9px;margin:3px 0 0 14px;">Online</div>
                    </div>

                    <div>
                        <div style="display:flex;align-items:center;gap:8px;color:#8B929C;font-size:11px;">
                            <span style="width:6px;height:6px;border-radius:50%;background:{LOW};"></span>
                            Local LLM (Ollama)
                        </div>
                        <div style="color:{LOW};font-family:'JetBrains Mono',monospace;font-size:9px;margin:3px 0 0 14px;">Online</div>
                    </div>
                </div>
            </div>

            <div style="
                color:#4D535C;
                font-family:'JetBrains Mono',monospace;
                font-size:9px;
                margin-top:48px;
            ">
                v0.1.0
            </div>
            """
        )

    return selected


# ---------------------------------------------------------------------------
# Command Center
# ---------------------------------------------------------------------------

def render_command_center(
    df: pd.DataFrame,
    event: dict,
    report: str,
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
        now = datetime.now().strftime("%d %b %Y · %H:%M")
        render_html(
            f"""
            <div class="fs-refresh">
                Last refreshed<br>
                <strong>{now}</strong>
            </div>
            """
        )

    st.write("")

    filtered_search_df = apply_search(
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

    visible_df = filter_dataset(
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

        render_top_three(visible_df)

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

        render_incident_panel(
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

        render_alerts(
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

    render_activity_table(
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



# ---------------------------------------------------------------------------
# Machine Intelligence
# ---------------------------------------------------------------------------

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
        df.sort_values("unified_risk_score", ascending=False)["Machine ID"]
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

    selected_rows = df[df["Machine ID"].astype(str) == selected_machine]

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
        .sort_values("unified_risk_score", ascending=False)
        .head(12)
        .copy()
    )

    ranking_rows = ""

    for rank, (_, machine_row) in enumerate(ranking.iterrows(), start=1):
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


# ---------------------------------------------------------------------------
# Secondary-page dispatcher
# ---------------------------------------------------------------------------

def render_secondary_page(
    selected_page: str,
    df: pd.DataFrame,
    event: dict,
    report: str,
) -> None:
    if selected_page == "Machine Intelligence":
        render_machine_intelligence(df)
        return

    page_descriptions = {
        "Predictive Health": (
            "Predictive Health",
            "C-MAPSS-based remaining useful life and degradation analysis."
        ),
        "Anomaly Detection": (
            "Anomaly Detection",
            "Unsupervised detection and reconstruction-based anomaly analysis."
        ),
        "Explainability": (
            "Explainability",
            "SHAP-based evidence showing which features influence failure predictions."
        ),
        "Incident Intelligence": (
            "Incident Intelligence",
            "Evidence-grounded incident reports generated from integrated risk events."
        ),
        "Safety Copilot": (
            "Safety Copilot",
            "Retrieval-grounded safety assistance using the ForgeShield knowledge base."
        ),
        "Knowledge Base": (
            "Knowledge Base",
            "Safety procedures, incident records, and retrieval evidence."
        ),
        "Documentation": (
            "Documentation",
            "Research methodology, datasets, models, and proof-of-concept notes."
        ),
        "Settings": (
            "Settings",
            "Dashboard configuration and research proof-of-concept information."
        ),
    }

    title, description = page_descriptions.get(
        selected_page,
        (selected_page, "ForgeShield module.")
    )

    render_html(
        f"""
        <div class="fs-eyebrow">{selected_page.upper()}</div>
        <div class="fs-page-title">{title}</div>
        <div class="fs-page-subtitle">{description}</div>
        <div style="height:24px;"></div>

        <div class="fs-panel">
            <div class="fs-state">
                <div class="fs-state-mark">[ {selected_page[:2].upper()} ]</div>
                <div class="fs-state-title">Module ready</div>
                <div class="fs-state-copy">
                    This section is reserved for the next ForgeShield
                    research module. The Command Center and Machine
                    Intelligence layers are already connected to the
                    integrated POC outputs.
                </div>
            </div>
        </div>

        <div class="fs-footer">
            <span>FORGESHIELD · INDUSTRIAL SAFETY INTELLIGENCE</span>
            <span class="fs-footer-mono">
                Research Proof of Concept · v0.1.0
            </span>
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    selected_page = render_sidebar()

    try:
        risk_df = load_risk_table()
        metadata_df = load_test_metadata()

        df = prepare_data(
            risk_df,
            metadata_df,
        )

        event = load_demo_event()
        report = load_incident_report()

    except Exception as exc:

        render_html(
            f"""
            <div class="fs-eyebrow">FORGESHIELD</div>
            <div class="fs-page-title">Command Center</div>
            <div class="fs-error">
                <strong>Dashboard data unavailable.</strong><br><br>
                {str(exc)}
            </div>
            """
        )

        return

    if selected_page == "Command Center":
        render_command_center(
            df,
            event,
            report,
        )
    else:
        render_secondary_page(
            selected_page,
            df,
            event,
            report,
        )


if __name__ == "__main__":
    main()
