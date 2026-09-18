from __future__ import annotations

import pandas as pd

from dashboard.components.ui import (
    empty_state,
    render_html,
    risk_pill,
)
from dashboard.components.theme import (
    CRITICAL,
    HIGH,
    LOW,
    MEDIUM,
)


def render_activity_table(
    df: pd.DataFrame,
    limit: int = 14,
) -> None:

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


def render_incident_panel(
    event: dict,
    report: str,
) -> None:

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
        "Integrated event generated from theAI4I held-out test set. "
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
