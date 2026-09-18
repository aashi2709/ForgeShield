from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def render_html(
    html: str,
    unsafe_allow_html: bool = True,
) -> None:
    """
    Render trusted ForgeShield HTML.

    The unsafe_allow_html parameter is retained for compatibility
    with dashboard views that pass it explicitly.

    Streamlit's st.html() is used because ForgeShield dashboard
    components rely on custom HTML and CSS rendering.
    """

    st.html(html.strip())


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------

def risk_class(risk: str) -> str:
    return {
        "Low": "fs-risk-low",
        "Medium": "fs-risk-medium",
        "High": "fs-risk-high",
        "Critical": "fs-risk-critical",
    }.get(
        str(risk),
        "fs-risk-medium",
    )


def risk_pill(risk: str) -> str:
    return (
        f'<span class="fs-risk-pill {risk_class(risk)}">'
        f"{risk}"
        f"</span>"
    )


# ---------------------------------------------------------------------------
# Numeric formatting
# ---------------------------------------------------------------------------

def metric_number(
    value: int | float,
) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}"

    return f"{int(value):,}"


# ---------------------------------------------------------------------------
# KPI card
# ---------------------------------------------------------------------------

def build_kpi(
    label: str,
    value: int | float,
    delta: str,
    meta: str,
    critical: bool = False,
) -> str:
    value_class = (
        "fs-kpi-critical"
        if critical
        else ""
    )

    return f"""
    <div class="fs-kpi">

        <div class="fs-kpi-label">
            {label}
        </div>

        <div class="fs-kpi-value {value_class}">
            {metric_number(value)}
        </div>

        <div class="fs-kpi-delta">
            −&nbsp;&nbsp;{delta}
        </div>

        <div class="fs-kpi-meta">
            {meta}
        </div>

    </div>
    """


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

def empty_state(
    title: str,
    copy: str,
    mark: str = "[]",
) -> str:
    return f"""
    <div class="fs-state">

        <div class="fs-state-mark">
            {mark}
        </div>

        <div class="fs-state-title">
            {title}
        </div>

        <div class="fs-state-copy">
            {copy}
        </div>

    </div>
    """