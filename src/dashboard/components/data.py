from pathlib import Path
import json

import pandas as pd
import streamlit as st


def load_risk_table(risk_table_path: Path) -> pd.DataFrame:
    """Load and validate the integrated machine-risk table."""
    if not risk_table_path.exists():
        raise FileNotFoundError(
            f"Missing integration output: {risk_table_path}"
        )

    df = pd.read_csv(risk_table_path)

    expected = {
        "failure_probability",
        "anomaly_score",
        "unified_risk_score",
        "risk_band",
        "actual_failure",
    }

    missing = expected.difference(df.columns)

    if missing:
        raise ValueError(
            "Integrated risk table is missing columns: "
            + ", ".join(sorted(missing))
        )

    return df


def load_test_metadata(test_metadata_path: Path) -> pd.DataFrame:
    """Load AI4I test metadata used by the dashboard."""
    if not test_metadata_path.exists():
        return pd.DataFrame()

    return pd.read_csv(test_metadata_path)


def load_demo_event(event_path: Path) -> dict:
    """Load the current ForgeShield demonstration event."""
    if not event_path.exists():
        return {}

    return json.loads(event_path.read_text(encoding="utf-8"))


def load_incident_report(report_path: Path) -> str:
    """Load the generated incident report."""
    if not report_path.exists():
        return ""

    return report_path.read_text(encoding="utf-8")


def load_validation(validation_path: Path) -> dict:
    """Load incident-report citation validation results."""
    if not validation_path.exists():
        return {}

    return json.loads(
        validation_path.read_text(encoding="utf-8")
    )
