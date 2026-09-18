from __future__ import annotations

from pathlib import Path
import json

import numpy as np
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


def prepare_data(
    risk_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine model outputs with held-out AI4I metadata."""

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

            if (
                "Air temperature" in df.columns
                and "Process temperature" in df.columns
            ):
                air_temperature = pd.to_numeric(
                    df["Air temperature"],
                    errors="coerce",
                )

                process_temperature = pd.to_numeric(
                    df["Process temperature"],
                    errors="coerce",
                )

                df["Temperature Differential"] = (
                    process_temperature - air_temperature
                )

            if (
                "Rotational speed" in df.columns
                and "Torque" in df.columns
            ):
                rotational_speed = pd.to_numeric(
                    df["Rotational speed"],
                    errors="coerce",
                )

                torque = pd.to_numeric(
                    df["Torque"],
                    errors="coerce",
                )

                df["Mechanical Power"] = (
                    rotational_speed
                    * torque
                    * (2.0 * np.pi / 60.0)
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
    """Return the requested dashboard dataset view."""

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
    """Filter machines by ID, product type, or risk band."""

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
