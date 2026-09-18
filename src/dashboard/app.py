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

from dashboard.components.charts import (
    render_risk_donut,
    render_scatter,
)  # noqa: E402

from dashboard.components.sidebar import render_sidebar  # noqa: E402

from dashboard.components.overview import (
    render_activity_table as _render_activity_table,
    render_top_three as _render_top_three,
    render_incident_panel as _render_incident_panel,
    render_alerts as _render_alerts,
)  # noqa: E402

from dashboard.pages.command_center import (
    render_command_center as _render_command_center,
)  # noqa: E402

from dashboard.pages.machine_intelligence import (
    render_machine_intelligence as _render_machine_intelligence,
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

# ---------------------------------------------------------------------------
# Tables / rail
# ---------------------------------------------------------------------------

render_activity_table = _render_activity_table
render_top_three = _render_top_three
render_incident_panel = _render_incident_panel
render_alerts = _render_alerts


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Command Center
# ---------------------------------------------------------------------------

render_command_center = _render_command_center


# ---------------------------------------------------------------------------
# Machine Intelligence
# ---------------------------------------------------------------------------

_safe_float = __import__(
    "dashboard.pages.machine_intelligence",
    fromlist=["_safe_float"],
)._safe_float

_format_sensor = __import__(
    "dashboard.pages.machine_intelligence",
    fromlist=["_format_sensor"],
)._format_sensor

render_machine_intelligence = _render_machine_intelligence


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
            apply_search,
            filter_dataset,
            render_top_three,
            render_incident_panel,
            render_alerts,
            render_activity_table,
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
