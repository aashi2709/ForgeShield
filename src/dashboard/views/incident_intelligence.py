from pathlib import Path
import json
import re

import pandas as pd
import streamlit as st

from dashboard.components.theme import MUTED
from dashboard.components.ui import empty_state, render_html


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_DIR = PROJECT_ROOT / "reports" / "integration"


EVENT_PATH = INTEGRATION_DIR / "demo_event.json"
REPORT_PATH = INTEGRATION_DIR / "incident_report.md"
VALIDATION_PATH = INTEGRATION_DIR / "incident_report_validation.json"
RISK_TABLE_PATH = INTEGRATION_DIR / "integrated_risk_table.csv"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def _load_report() -> str:
    if not REPORT_PATH.exists():
        return ""

    try:
        return REPORT_PATH.read_text(encoding="utf-8")
    except Exception:
        return ""


def _load_risk_table() -> pd.DataFrame:
    if not RISK_TABLE_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(RISK_TABLE_PATH)
    except Exception:
        return pd.DataFrame()


def _metric_card(label: str, value: str, detail: str = "") -> str:
    return f"""
    <div class="fs-card" style="padding:18px 20px; min-height:108px;">
        <div style="
            font-size:11px;
            letter-spacing:0.12em;
            text-transform:uppercase;
            color:{MUTED};
            font-weight:700;
            margin-bottom:10px;
        ">
            {label}
        </div>

        <div style="
            font-size:27px;
            font-weight:750;
            color:#F5F7FA;
            line-height:1.1;
        ">
            {value}
        </div>

        <div style="
            font-size:12px;
            color:{MUTED};
            margin-top:7px;
        ">
            {detail}
        </div>
    </div>
    """


def _risk_class(band: str) -> str:
    return {
        "Critical": "#FF5C5C",
        "High": "#FF9F43",
        "Medium": "#FFD166",
        "Low": "#55D187",
    }.get(str(band), "#AAB2BF")


def _extract_citations(report: str) -> list[str]:
    """Extract the evidence IDs used in the generated report."""
    ids = re.findall(r"\b(?:INC-\d{4}|FSP-SOP-\d{3})\b", report)
    return list(dict.fromkeys(ids))


def _section_text(report: str, heading: str) -> str:
    """Extract one markdown heading section without changing its content."""
    if not report:
        return ""

    pattern = rf"(?ms)^##\s+{re.escape(heading)}\s*$\n(.*?)(?=^##\s+|\Z)"
    match = re.search(pattern, report)

    if not match:
        return ""

    return match.group(1).strip()


def render_incident_intelligence() -> None:
    """Render the ForgeShield evidence-grounded incident intelligence view."""

    st.markdown(
        """
        <div class="fs-page-kicker">INCIDENT INTELLIGENCE</div>

        <div class="fs-page-title">
            Evidence-Grounded Incident Analysis
        </div>

        <div class="fs-page-subtitle">
            Integrated risk signals, retrieved safety evidence, and validated
            AI-generated incident reporting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    event = _load_json(EVENT_PATH)
    report = _load_report()
    validation = _load_json(VALIDATION_PATH)
    risk_df = _load_risk_table()

    if not event and not report:
        empty_state(
            "Incident intelligence artifacts are not available.",
            "Run the integrated ForgeShield risk and incident-report pipeline first.",
        )
        return

    risk_score = float(event.get("risk_score", 0.0))
    failure_probability = float(event.get("failure_probability", 0.0))
    anomaly_score = float(event.get("anomaly_score", 0.0))
    risk_band = str(event.get("risk_band", "Unknown"))

    citations = _extract_citations(report)

    validation_status = validation.get("valid", validation.get("validation_passed"))
    if validation_status is True:
        validation_label = "Validated"
    elif validation_status is False:
        validation_label = "Review"
    else:
        validation_label = "Available"

    render_html(
        f"""
        <div style="
            display:grid;
            grid-template-columns:repeat(4, 1fr);
            gap:14px;
            margin:18px 0 26px 0;
        ">
            {_metric_card(
                "Unified risk",
                f"{risk_score:.3f}",
                f"{risk_band} risk band"
            )}

            {_metric_card(
                "Failure probability",
                f"{failure_probability:.3f}",
                "Supervised model signal"
            )}

            {_metric_card(
                "Anomaly score",
                f"{anomaly_score:.3f}",
                "Isolation Forest signal"
            )}

            {_metric_card(
                "Report status",
                validation_label,
                f"{len(citations)} cited evidence IDs"
            )}
        </div>
        """
    )

    # ------------------------------------------------------------------
    # Incident signal
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Incident Signal
        </div>

        <div class="fs-section-subtitle">
            The incident report is generated from the integrated POC event,
            preserving the underlying model signals as structured evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    event_cols = st.columns(4)

    event_items = [
        ("Machine", str(event.get("machine_id", "Unknown"))),
        ("Event type", str(event.get("event_type", "Unknown"))),
        ("Product type", str(event.get("product_type", "Unknown"))),
        ("Actual failure", str(event.get("actual_failure", "Unknown"))),
    ]

    for column, (label, value) in zip(event_cols, event_items):
        with column:
            st.markdown(
                f"""
                <div class="fs-card" style="padding:16px 18px;">
                    <div style="
                        color:{MUTED};
                        font-size:11px;
                        text-transform:uppercase;
                        letter-spacing:0.1em;
                        font-weight:700;
                    ">
                        {label}
                    </div>
                    <div style="
                        margin-top:8px;
                        font-size:18px;
                        font-weight:700;
                    ">
                        {value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    sensor_fields = [
        ("Air temperature", "air_temperature", "K"),
        ("Process temperature", "process_temperature", "K"),
        ("Rotational speed", "rotational_speed", "rpm"),
        ("Torque", "torque", "Nm"),
        ("Tool wear", "tool_wear", "min"),
        ("Temperature differential", "temperature_differential", "K"),
        ("Mechanical power", "mechanical_power", "W"),
    ]

    sensor_rows = []

    for label, key, unit in sensor_fields:
        if key in event:
            sensor_rows.append(
                {
                    "Sensor": label,
                    "Value": event[key],
                    "Unit": unit,
                }
            )

    if sensor_rows:
        sensor_df = pd.DataFrame(sensor_rows)
        st.dataframe(
            sensor_df,
            use_container_width=True,
            hide_index=True,
        )

    # ------------------------------------------------------------------
    # Risk composition
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Risk Composition
        </div>
        """,
        unsafe_allow_html=True,
    )

    risk_components = pd.DataFrame(
        {
            "Signal": [
                "Failure probability",
                "Anomaly score",
                "Unified risk score",
            ],
            "Score": [
                failure_probability,
                anomaly_score,
                risk_score,
            ],
        }
    )

    st.bar_chart(
        risk_components.set_index("Signal"),
        y="Score",
        height=250,
    )

    st.caption(
        "The unified risk score combines the supervised failure probability "
        "and continuous anomaly score using the ForgeShield POC risk-scoring layer."
    )

    # ------------------------------------------------------------------
    # Generated incident report
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Generated Incident Report
        </div>

        <div class="fs-section-subtitle">
            The report below is the persisted output of the evidence-grounded
            incident-report generation pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if report:
        st.markdown(
            f"""
            <div class="fs-card" style="
                padding:24px;
                border-left:3px solid {_risk_class(risk_band)};
            ">
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(report)

    # ------------------------------------------------------------------
    # Evidence and validation
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Evidence & Validation
        </div>

        <div class="fs-section-subtitle">
            Retrieved evidence identifiers are checked against the persisted
            validation artifact before the report is treated as grounded.
        </div>
        """,
        unsafe_allow_html=True,
    )

    evidence_cols = st.columns(3)

    with evidence_cols[0]:
        render_html(
            _metric_card(
                "Evidence citations",
                str(len(citations)),
                "Unique IDs found in generated report",
            )
        )

    with evidence_cols[1]:
        render_html(
            _metric_card(
                "Validation",
                validation_label,
                "Persisted incident-report validation result",
            )
        )

    with evidence_cols[2]:
        expected = validation.get("expected_citations")
        if isinstance(expected, list):
            expected_value = str(len(expected))
        else:
            expected_value = "Available"

        render_html(
            _metric_card(
                "Expected evidence",
                expected_value,
                "Recorded by validation artifact",
            )
        )

    if citations:
        st.markdown("**Cited evidence identifiers**")
        st.dataframe(
            pd.DataFrame({"Evidence ID": citations}),
            use_container_width=True,
            hide_index=True,
        )

    if validation:
        with st.expander("Validation artifact"):
            st.json(validation)

    # ------------------------------------------------------------------
    # Integrated risk context
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Integrated Risk Context
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not risk_df.empty:
        display_columns = [
            column
            for column in [
                "machine_id",
                "event_type",
                "risk_score",
                "risk_band",
                "failure_probability",
                "anomaly_score",
                "actual_failure",
            ]
            if column in risk_df.columns
        ]

        if display_columns:
            context_df = risk_df[display_columns].copy()

            if "risk_score" in context_df.columns:
                context_df = context_df.sort_values(
                    "risk_score",
                    ascending=False,
                ).head(10)

            st.dataframe(
                context_df,
                use_container_width=True,
                hide_index=True,
            )

    # ------------------------------------------------------------------
    # Research framing
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Research Contribution
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fs-card" style="padding:22px 24px; line-height:1.7;">

        <strong>Unified decision layer</strong><br>
        ForgeShield connects supervised failure probability with an
        unsupervised anomaly signal to produce a single interpretable
        risk representation.

        <br><br>

        <strong>Evidence-grounded generation</strong><br>
        The incident-reporting layer uses retrieved synthetic research
        incidents and safety procedures as grounding evidence for the
        generated report.

        <br><br>

        <strong>Explicit validation</strong><br>
        Generated evidence identifiers are checked against the retrieved
        evidence set, providing a lightweight validation layer for the
        proof-of-concept workflow.

        <br><br>

        <strong>Research limitation</strong><br>
        The current incident corpus is synthetic research evidence.
        Therefore, generated recommendations demonstrate the architecture
        and grounding workflow rather than validated operational safety
        procedures for deployment.

        </div>
        """,
        unsafe_allow_html=True,
    )

    render_html(
        """
        <div style="
            margin-top:24px;
            padding-top:18px;
            border-top:1px solid rgba(255,255,255,0.08);
            color:#69717F;
            font-size:12px;
            display:flex;
            justify-content:space-between;
        ">
            <span>FORGESHIELD · INCIDENT INTELLIGENCE</span>
            <span>Research Proof of Concept · Evidence-Grounded Reporting</span>
        </div>
        """
    )
