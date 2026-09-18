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

from dashboard.views.live_monitoring import (
    render_live_monitoring,
)

from dashboard.views.command_center import (
    render_command_center as _render_command_center,
)  # noqa: E402

from dashboard.views.machine_intelligence import (
    render_machine_intelligence as _render_machine_intelligence,
)  # noqa: E402

from dashboard.views.predictive_health import (
    render_predictive_health as _render_predictive_health,
)  # noqa: E402

from dashboard.views.anomaly_detection import (
    render_anomaly_detection as _render_anomaly_detection,
)  # noqa: E402

from dashboard.views.explainability import (
    render_explainability as _render_explainability,
)  # noqa: E402

from dashboard.views.incident_intelligence import (
    render_incident_intelligence as _render_incident_intelligence,
)  # noqa: E402

from dashboard.views.safety_copilot import (
    render_safety_copilot as _render_safety_copilot,
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
render_anomaly_detection = _render_anomaly_detection
render_explainability = _render_explainability
render_incident_intelligence = _render_incident_intelligence
render_safety_copilot = _render_safety_copilot


# ---------------------------------------------------------------------------
# Secondary-page dispatcher
# ---------------------------------------------------------------------------

def render_secondary_page(
    selected_page: str,
    df: pd.DataFrame,
    event: dict,
    report: str,
) -> None:

    if selected_page == "Live Monitoring":
        render_live_monitoring()
        return

    if selected_page == "Machine Intelligence":
        render_machine_intelligence(df)
        return

    if selected_page == "Predictive Health":
        render_predictive_health()
        return

    if selected_page == "Anomaly Detection":
        render_anomaly_detection()
        return

    if selected_page == "Explainability":
        render_explainability()
        return

    if selected_page == "Incident Intelligence":
        render_incident_intelligence()
        return

    if selected_page == "Safety Copilot":
        render_safety_copilot()
        return

    if selected_page == "Knowledge Base":
        render_html(
            """
            <div class="fs-eyebrow">KNOWLEDGE BASE</div>
            <div class="fs-page-title">ForgeShield Research Knowledge Base</div>
            <div class="fs-page-subtitle">
                Retrieval corpus supporting evidence-grounded safety analysis.
            </div>
            <div style="height:24px;"></div>
            """
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Documents", "84")

        with col2:
            st.metric("Indexed Chunks", "158")

        with col3:
            st.metric("Vector Store", "ChromaDB")

        st.markdown("### Evidence Coverage")
        st.markdown(
            """
            - **Safety procedures:** 4
            - **Synthetic incident records:** 80
            - **Incident types:** 8
            - **Embedding dimension:** 384
            """
        )

        st.markdown("### Evidence Policy")
        st.info(
            "The current research corpus contains synthetic safety-event records "
            "and synthetic research procedures. These artifacts demonstrate the "
            "RAG pipeline and must not be interpreted as validated industrial "
            "operating procedures."
        )

        st.markdown("### Retrieval Pipeline")
        st.markdown(
            "Question → Embedding → ChromaDB retrieval → Evidence filtering → "
            "Grounded generation → Citation validation"
        )

        return

    if selected_page == "Documentation":
        render_html(
            """
            <div class="fs-eyebrow">DOCUMENTATION</div>
            <div class="fs-page-title">Research Documentation</div>
            <div class="fs-page-subtitle">
                ForgeShield proof-of-concept methodology, datasets, models, and
                research scope.
            </div>
            <div style="height:24px;"></div>
            """
        )

        st.markdown("### Research Pipeline")
        st.markdown(
            """
            **Industrial Data → Data Engineering → EDA → ML → Anomaly Detection
            → Deep Learning → Risk Prediction → XAI → RAG → GenAI Safety Copilot**
            """
        )

        st.markdown("### Datasets")
        st.markdown(
            """
            | Dataset | Purpose |
            |---|---|
            | **AI4I 2020** | Machine failure classification |
            | **NASA C-MAPSS FD001** | Remaining Useful Life prediction |
            | **Synthetic Safety Events** | Risk integration and RAG demonstration |
            """
        )

        st.markdown("### Model Families")
        st.markdown(
            """
            - Logistic Regression
            - Decision Tree
            - Random Forest
            - KNN
            - SVM
            - Naive Bayes
            - Gradient Boosting
            - XGBoost
            - LSTM
            - GRU
            - CNN-LSTM
            - LSTM Autoencoder
            - Isolation Forest
            - One-Class SVM
            - K-Means
            - DBSCAN
            - Agglomerative Clustering
            """
        )

        st.markdown("### Explainability & GenAI")
        st.markdown(
            """
            **XAI:** SHAP-based global and model-specific feature attribution.

            **RAG:** Evidence retrieval from the ForgeShield research knowledge
            base followed by constrained local LLM generation.

            **Safety Copilot:** Separates facts, hypotheses, recommended actions,
            evidence sources, and limitations.
            """
        )

        st.markdown("### Research Scope")
        st.info(
            "ForgeShield is a research proof of concept. The system demonstrates "
            "the proposed analytical and generative-AI architecture rather than "
            "a production-certified industrial safety system."
        )

        return

    if selected_page == "Settings":
        render_html(
            """
            <div class="fs-eyebrow">SETTINGS</div>
            <div class="fs-page-title">ForgeShield Configuration</div>
            <div class="fs-page-subtitle">
                Runtime configuration and proof-of-concept system status.
            </div>
            <div style="height:24px;"></div>
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### AI Runtime")
            st.markdown(
                """
                **LLM:** Qwen3:8B  
                **Inference:** Local Ollama  
                **Generation:** Evidence-constrained
                """
            )

        with col2:
            st.markdown("### Retrieval")
            st.markdown(
                """
                **Vector Store:** ChromaDB  
                **Knowledge Base:** ForgeShield research corpus  
                **Indexed Chunks:** 158
                """
            )

        st.markdown("### System Status")
        status_rows = [
            {"Component": "Supervised Models", "Status": "Ready"},
            {"Component": "Anomaly Detection", "Status": "Ready"},
            {"Component": "C-MAPSS RUL Models", "Status": "Ready"},
            {"Component": "SHAP Explainability", "Status": "Ready"},
            {"Component": "Risk Scoring", "Status": "Ready"},
            {"Component": "RAG Retrieval", "Status": "Ready"},
            {"Component": "Safety Copilot", "Status": "Ready"},
        ]

        st.dataframe(
            status_rows,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Research Mode")
        st.info(
            "ForgeShield is configured as a local research proof of concept. "
            "Synthetic evidence is explicitly identified and should not be "
            "treated as validated industrial safety guidance."
        )

        return

    page_descriptions = {
        "Predictive Health": (
            "Predictive Health",
            "C-MAPSS-based remaining useful life and degradation analysis.",
        ),
        "Anomaly Detection": (
            "Anomaly Detection",
            "Unsupervised detection and reconstruction-based anomaly analysis.",
        ),
        "Explainability": (
            "Explainability",
            "SHAP-based evidence showing which features influence failure predictions.",
        ),
        "Incident Intelligence": (
            "Incident Intelligence",
            "Evidence-grounded incident reports generated from integrated risk events.",
        ),
        "Safety Copilot": (
            "Safety Copilot",
            "Retrieval-grounded safety assistance using the ForgeShield knowledge base.",
        ),
        "Knowledge Base": (
            "Knowledge Base",
            "Safety procedures, incident records, and retrieval evidence.",
        ),
        "Documentation": (
            "Documentation",
            "Research methodology, datasets, models, and proof-of-concept notes.",
        ),
        "Settings": (
            "Settings",
            "Dashboard configuration and research proof-of-concept information.",
        ),
    }

    title, description = page_descriptions.get(
        selected_page,
        (selected_page, "ForgeShield module."),
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