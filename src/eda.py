from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i" / "ai4i2020.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "ai4i"


NUMERICAL_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

FAILURE_MODES = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


def load_data() -> pd.DataFrame:
    """Load the raw AI4I dataset."""

    return pd.read_csv(DATA_PATH)


def setup_output() -> None:
    """Create the EDA output directory."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(filename: str) -> None:
    """Save the current figure at publication-friendly resolution."""

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# ---------------------------------------------------------------------
# BASIC EDA
# ---------------------------------------------------------------------

def class_distribution(df: pd.DataFrame) -> None:
    counts = df["Machine failure"].value_counts().sort_index()

    labels = ["Normal", "Failure"]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=labels, y=counts.values)

    plt.title("Machine Failure Distribution")
    plt.xlabel("Machine State")
    plt.ylabel("Number of Observations")

    save_plot("01_machine_failure_distribution.png")


def failure_mode_distribution(df: pd.DataFrame) -> None:
    counts = [df[mode].sum() for mode in FAILURE_MODES]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=FAILURE_MODES, y=counts)

    plt.title("Failure Mode Distribution")
    plt.xlabel("Failure Mode")
    plt.ylabel("Number of Observations")

    save_plot("02_failure_mode_distribution.png")


def failure_by_type(df: pd.DataFrame) -> None:
    failure_rate = (
        df.groupby("Type")["Machine failure"]
        .mean()
        .mul(100)
        .reset_index(name="Failure Rate (%)")
    )

    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=failure_rate,
        x="Type",
        y="Failure Rate (%)",
    )

    plt.title("Machine Failure Rate by Product Type")
    plt.xlabel("Product Type")
    plt.ylabel("Failure Rate (%)")

    save_plot("03_failure_rate_by_type.png")


def numerical_distributions(df: pd.DataFrame) -> None:
    for feature in NUMERICAL_FEATURES:

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=feature,
            kde=True,
        )

        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")

        filename = (
            feature.lower()
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
        )

        save_plot(f"04_distribution_{filename}.png")


def feature_failure_comparison(df: pd.DataFrame) -> None:
    for feature in NUMERICAL_FEATURES:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="Machine failure",
            y=feature,
        )

        plt.title(f"{feature}: Normal vs Failure")
        plt.xlabel("Machine Failure (0 = Normal, 1 = Failure)")
        plt.ylabel(feature)

        filename = (
            feature.lower()
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
        )

        save_plot(f"05_failure_comparison_{filename}.png")


def correlation_analysis(df: pd.DataFrame) -> None:
    numerical_df = df.select_dtypes(include="number")

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        numerical_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
    )

    plt.title("AI4I Numerical Feature Correlation Matrix")

    save_plot("06_correlation_heatmap.png")


# ---------------------------------------------------------------------
# RESEARCH-FOCUSED EDA
# ---------------------------------------------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create candidate features for exploratory analysis only.

    The original dataframe is not modified.
    """

    data = df.copy()

    data["Temperature Differential"] = (
        data["Process temperature"]
        - data["Air temperature"]
    )

    # Mechanical power proxy:
    # Power ∝ Torque × Angular Velocity
    # Angular velocity = RPM × 2π / 60
    data["Mechanical Power"] = (
        data["Torque"]
        * data["Rotational speed"]
        * (2 * 3.141592653589793 / 60)
    )

    return data


def failure_rate_by_bins(
    df: pd.DataFrame,
    feature: str,
    filename: str,
    title: str,
) -> None:
    """
    Calculate failure rate across quantile-based bins.

    Quantile bins ensure that each group contains a
    reasonably comparable number of observations.
    """

    data = df.copy()

    data["bin"] = pd.qcut(
        data[feature],
        q=10,
        duplicates="drop",
    )

    grouped = (
        data.groupby("bin", observed=True)["Machine failure"]
        .mean()
        .mul(100)
        .reset_index(name="Failure Rate (%)")
    )

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=grouped,
        x="bin",
        y="Failure Rate (%)",
    )

    plt.title(title)
    plt.xlabel(feature + " decile")
    plt.ylabel("Failure Rate (%)")
    plt.xticks(rotation=45)

    save_plot(filename)


def operating_range_failure_analysis(df: pd.DataFrame) -> None:
    """
    Examine how observed failure rates change across
    operating-variable ranges.
    """

    failure_rate_by_bins(
        df,
        "Torque",
        "07_failure_rate_by_torque_range.png",
        "Failure Rate Across Torque Ranges",
    )

    failure_rate_by_bins(
        df,
        "Rotational speed",
        "08_failure_rate_by_speed_range.png",
        "Failure Rate Across Rotational Speed Ranges",
    )

    failure_rate_by_bins(
        df,
        "Tool wear",
        "09_failure_rate_by_tool_wear_range.png",
        "Failure Rate Across Tool Wear Ranges",
    )


def engineered_feature_analysis(df: pd.DataFrame) -> None:
    """Analyze candidate engineered features."""

    data = add_engineered_features(df)

    failure_rate_by_bins(
        data,
        "Temperature Differential",
        "10_failure_rate_by_temperature_differential.png",
        "Failure Rate Across Temperature Differential",
    )

    failure_rate_by_bins(
        data,
        "Mechanical Power",
        "11_failure_rate_by_mechanical_power.png",
        "Failure Rate Across Mechanical Power",
    )


def failure_mode_cooccurrence(df: pd.DataFrame) -> None:
    """
    Examine relationships among the five documented
    failure-mode indicators.
    """

    correlation = df[FAILURE_MODES].corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
    )

    plt.title("Failure Mode Co-occurrence Correlation")

    save_plot("12_failure_mode_correlation.png")


def feature_failure_association(df: pd.DataFrame) -> None:
    """
    Calculate Pearson correlation between numerical
    candidate features and the machine-failure target.
    """

    data = add_engineered_features(df)

    features = NUMERICAL_FEATURES + [
        "Temperature Differential",
        "Mechanical Power",
    ]

    correlations = (
        data[features + ["Machine failure"]]
        .corr()["Machine failure"]
        .drop("Machine failure")
        .sort_values(key=abs, ascending=False)
    )

    summary = correlations.reset_index()

    summary.columns = [
        "Feature",
        "Correlation with Machine Failure",
    ]

    summary.to_csv(
        FIGURE_DIR / "feature_failure_correlations.csv",
        index=False,
    )

    print("\n[RESEARCH] Feature association with machine failure")
    print(summary.to_string(index=False))


def failure_rate_summary(df: pd.DataFrame) -> None:
    """Print failure rates by important categorical variables."""

    print("\n[RESEARCH] Failure rate by product type")

    type_summary = (
        df.groupby("Type")["Machine failure"]
        .agg(["count", "sum", "mean"])
    )

    type_summary["failure_rate_percent"] = (
        type_summary["mean"] * 100
    )

    print(type_summary.to_string())

    print("\n[RESEARCH] Failure-mode counts")

    print(
        df[FAILURE_MODES]
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )


def print_engineered_feature_summary(df: pd.DataFrame) -> None:
    """Print descriptive statistics for candidate engineered features."""

    data = add_engineered_features(df)

    engineered = [
        "Temperature Differential",
        "Mechanical Power",
    ]

    print("\n[RESEARCH] Candidate engineered features")

    print(
        data[engineered]
        .describe()
        .T
        .to_string()
    )


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def run_eda() -> None:
    """Run the complete AI4I exploratory analysis."""

    print("\n" + "=" * 60)
    print("FORGESHIELD | AI4I EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = load_data()

    print(f"\nDataset shape: {df.shape}")

    setup_output()

    print("\nGenerating basic EDA...")

    class_distribution(df)
    failure_mode_distribution(df)
    failure_by_type(df)
    numerical_distributions(df)
    feature_failure_comparison(df)
    correlation_analysis(df)

    print("Basic EDA complete.")

    print("\nGenerating research-focused analysis...")

    operating_range_failure_analysis(df)
    engineered_feature_analysis(df)
    failure_mode_cooccurrence(df)
    feature_failure_association(df)

    failure_rate_summary(df)
    print_engineered_feature_summary(df)

    print("\n" + "=" * 60)
    print("AI4I EDA COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()