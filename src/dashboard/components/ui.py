from __future__ import annotations


def render_html(html: str) -> None:
    import streamlit as st

    st.html(html.strip())


def risk_class(risk: str) -> str:
    return {
        "Low": "fs-risk-low",
        "Medium": "fs-risk-medium",
        "High": "fs-risk-high",
        "Critical": "fs-risk-critical",
    }.get(str(risk), "fs-risk-medium")


def risk_pill(risk: str) -> str:
    return (
        f'<span class="fs-risk-pill {risk_class(risk)}">'
        f"{risk}"
        f"</span>"
    )


def metric_number(value: int | float) -> str:
    if isinstance(value, float) and not value.is_integer():
        return f"{value:.3f}"

    return f"{int(value):,}"


def build_kpi(
    label: str,
    value: int | float,
    delta: str,
    meta: str,
    critical: bool = False,
) -> str:

    value_class = "fs-kpi-critical" if critical else ""

    return f"""
    <div class="fs-kpi">
        <div class="fs-kpi-label">{label}</div>
        <div class="fs-kpi-value {value_class}">
            {metric_number(value)}
        </div>
        <div class="fs-kpi-delta">−&nbsp;&nbsp;{delta}</div>
        <div class="fs-kpi-meta">{meta}</div>
    </div>
    """


def empty_state(
    title: str,
    copy: str,
    mark: str = "[]",
) -> str:

    return f"""
    <div class="fs-state">
        <div class="fs-state-mark">{mark}</div>
        <div class="fs-state-title">{title}</div>
        <div class="fs-state-copy">{copy}</div>
    </div>
    """
