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
    prepare_data as _prepare_data,
    filter_dataset as _filter_dataset,
    apply_search as _apply_search,
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

from dashboard.views.command_center import (
    render_command_center as _render_command_center,
)  # noqa: E402

from dashboard.views.machine_intelligence import (
    render_machine_intelligence as _render_machine_intelligence,
)  # noqa: E402

from dashboard.views.predictive_health import (
    render_predictive_health as _render_predictive_health,
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

prepare_data = _prepare_data
filter_dataset = _filter_dataset
apply_search = _apply_search


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
    "dashboard.views.machine_intelligence",
    fromlist=["_safe_float"],
)._safe_float

_format_sensor = __import__(
    "dashboard.views.machine_intelligence",
    fromlist=["_format_sensor"],
)._format_sensor

render_machine_intelligence = _render_machine_intelligence
render_predictive_health = _render_predictive_health


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

    if selected_page == "Predictive Health":
        render_predictive_health()
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
