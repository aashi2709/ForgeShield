from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

AI4I_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ai4i"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "supervised"
)

ANOMALY_MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "anomaly"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "risk_scoring"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "risk"
)

RANDOM_STATE = 42

# Default research-PoC weights from the ForgeShield specification.
FAILURE_WEIGHT = 0.60
ANOMALY_WEIGHT = 0.40


def load_ai4i_data():

    X_train = pd.read_csv(
        AI4I_DIR / "X_train.csv"
    )

    X_test = pd.read_csv(
        AI4I_DIR / "X_test.csv"
    )

    y_train = pd.read_csv(
        AI4I_DIR / "y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        AI4I_DIR / "y_test.csv"
    ).squeeze()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def load_supervised_model():

    # Gradient Boosting is used here because it had
    # strong cross-validation performance and produces
    # probability estimates directly.

    candidates = [
        MODEL_DIR / "gradient_boosting.joblib",
        MODEL_DIR / "gradient_boosting.pkl",
        MODEL_DIR / "Gradient Boosting.joblib",
        MODEL_DIR / "Gradient_Boosting.joblib",
    ]

    for path in candidates:

        if path.exists():

            print(
                f"\nLoading supervised model:"
            )

            print(
                f"  {path}"
            )

            return joblib.load(path)

    raise FileNotFoundError(
        "Gradient Boosting model not found "
        f"in {MODEL_DIR}"
    )


def load_isolation_forest():

    candidates = [
        ANOMALY_MODEL_DIR
        / "isolation_forest.joblib",

        ANOMALY_MODEL_DIR
        / "isolation_forest.pkl",
    ]

    for path in candidates:

        if path.exists():

            print(
                "\nLoading anomaly detector:"
            )

            print(
                f"  {path}"
            )

            return joblib.load(path)

    raise FileNotFoundError(
        "Isolation Forest model not found "
        f"in {ANOMALY_MODEL_DIR}"
    )


def min_max_normalize(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:

        return np.zeros_like(
            values
        )

    return (
        (values - minimum)
        / (maximum - minimum)
    )


def calculate_anomaly_score(
    isolation_forest,
    X,
):

    # Isolation Forest's decision_function:
    #
    # larger value  -> more normal
    # smaller value -> more anomalous
    #
    # We invert the values so that:
    #
    # larger score -> higher anomaly risk

    decision_scores = (
        isolation_forest.decision_function(
            X
        )
    )

    anomaly_score = (
        -decision_scores
    )

    anomaly_score = (
        min_max_normalize(
            anomaly_score
        )
    )

    return anomaly_score


def calculate_risk_band(
    risk_score,
):

    if risk_score < 0.25:

        return "Low"

    elif risk_score < 0.50:

        return "Medium"

    elif risk_score < 0.75:

        return "High"

    else:

        return "Critical"


def calculate_risk_scores(
    failure_probability,
    anomaly_score,
):

    unified_risk = (
        FAILURE_WEIGHT
        * failure_probability
        +
        ANOMALY_WEIGHT
        * anomaly_score
    )

    risk_bands = [
        calculate_risk_band(
            score
        )
        for score in unified_risk
    ]

    return (
        unified_risk,
        risk_bands,
    )


def create_risk_distribution(
    risk_df,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        risk_df[
            "unified_risk_score"
        ],
        bins=40,
        alpha=0.75,
    )

    plt.axvline(
        0.25,
        linestyle="--",
        linewidth=1.5,
        label="Low → Medium",
    )

    plt.axvline(
        0.50,
        linestyle="--",
        linewidth=1.5,
        label="Medium → High",
    )

    plt.axvline(
        0.75,
        linestyle="--",
        linewidth=1.5,
        label="High → Critical",
    )

    plt.xlabel(
        "Unified Risk Score"
    )

    plt.ylabel(
        "Number of Observations"
    )

    plt.title(
        "ForgeShield Unified Risk Score Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "unified_risk_distribution.png",
        dpi=300,
    )

    plt.close()


def create_risk_band_distribution(
    risk_df,
):

    band_order = [
        "Low",
        "Medium",
        "High",
        "Critical",
    ]

    counts = (
        risk_df[
            "risk_band"
        ]
        .value_counts()
        .reindex(
            band_order,
            fill_value=0,
        )
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        counts.index,
        counts.values,
    )

    plt.xlabel(
        "Risk Band"
    )

    plt.ylabel(
        "Number of Observations"
    )

    plt.title(
        "ForgeShield Risk Band Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "risk_band_distribution.png",
        dpi=300,
    )

    plt.close()


def create_risk_vs_failure_plot(
    risk_df,
):

    normal = risk_df[
        risk_df["actual_failure"] == 0
    ]

    failure = risk_df[
        risk_df["actual_failure"] == 1
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.scatter(
        normal.index,
        normal[
            "unified_risk_score"
        ],
        alpha=0.35,
        s=12,
        label="Normal",
    )

    plt.scatter(
        failure.index,
        failure[
            "unified_risk_score"
        ],
        alpha=0.75,
        s=25,
        label="Known Failure",
    )

    plt.axhline(
        0.25,
        linestyle="--",
        linewidth=1.2,
    )

    plt.axhline(
        0.50,
        linestyle="--",
        linewidth=1.2,
    )

    plt.axhline(
        0.75,
        linestyle="--",
        linewidth=1.2,
    )

    plt.xlabel(
        "Test Observation"
    )

    plt.ylabel(
        "Unified Risk Score"
    )

    plt.title(
        "Unified Risk Score vs. Known Machine Failure"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "risk_vs_failure.png",
        dpi=300,
    )

    plt.close()


def print_summary(
    risk_df,
):

    print(
        "\n" + "-" * 70
    )

    print(
        "RISK SCORE SUMMARY"
    )

    print(
        f"  Mean risk score: "
        f"{risk_df['unified_risk_score'].mean():.4f}"
    )

    print(
        f"  Median risk score: "
        f"{risk_df['unified_risk_score'].median():.4f}"
    )

    print(
        f"  Minimum risk score: "
        f"{risk_df['unified_risk_score'].min():.4f}"
    )

    print(
        f"  Maximum risk score: "
        f"{risk_df['unified_risk_score'].max():.4f}"
    )

    print(
        "\nRisk bands:"
    )

    band_counts = (
        risk_df[
            "risk_band"
        ]
        .value_counts()
        .reindex(
            [
                "Low",
                "Medium",
                "High",
                "Critical",
            ],
            fill_value=0,
        )
    )

    for band, count in (
        band_counts.items()
    ):

        percentage = (
            100
            * count
            / len(risk_df)
        )

        print(
            f"  {band:8s}: "
            f"{count:4d} "
            f"({percentage:.2f}%)"
        )


def main():

    print("\n" + "=" * 70)

    print(
        "FORGESHIELD | UNIFIED RISK SCORING"
    )

    print("=" * 70)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print(
        "\nLoading AI4I data..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_ai4i_data()

    print(
        f"  X_train: {X_train.shape}"
    )

    print(
        f"  X_test : {X_test.shape}"
    )

    print(
        f"  Test failures: "
        f"{y_test.sum()}"
    )

    # ---------------------------------------------------------
    # Load models
    # ---------------------------------------------------------

    supervised_model = (
        load_supervised_model()
    )

    isolation_forest = (
        load_isolation_forest()
    )

    # ---------------------------------------------------------
    # Failure probability
    # ---------------------------------------------------------

    print(
        "\nCalculating supervised failure probability..."
    )

    failure_probability = (
        supervised_model.predict_proba(
            X_test
        )[:, 1]
    )

    print(
        f"  Mean failure probability: "
        f"{failure_probability.mean():.4f}"
    )

    print(
        f"  Maximum failure probability: "
        f"{failure_probability.max():.4f}"
    )

    # ---------------------------------------------------------
    # Anomaly score
    # ---------------------------------------------------------

    print(
        "\nCalculating normalized anomaly score..."
    )

    anomaly_score = (
        calculate_anomaly_score(
            isolation_forest,
            X_test,
        )
    )

    print(
        f"  Mean anomaly score: "
        f"{anomaly_score.mean():.4f}"
    )

    print(
        f"  Maximum anomaly score: "
        f"{anomaly_score.max():.4f}"
    )

    # ---------------------------------------------------------
    # Unified risk
    # ---------------------------------------------------------

    print(
        "\nCombining risk components..."
    )

    print(
        f"  Failure probability weight: "
        f"{FAILURE_WEIGHT:.2f}"
    )

    print(
        f"  Anomaly score weight: "
        f"{ANOMALY_WEIGHT:.2f}"
    )

    (
        unified_risk,
        risk_bands,
    ) = calculate_risk_scores(
        failure_probability,
        anomaly_score,
    )

    # ---------------------------------------------------------
    # Build risk table
    # ---------------------------------------------------------

    risk_df = X_test.copy()

    risk_df[
        "failure_probability"
    ] = failure_probability

    risk_df[
        "anomaly_score"
    ] = anomaly_score

    risk_df[
        "unified_risk_score"
    ] = unified_risk

    risk_df[
        "risk_band"
    ] = risk_bands

    risk_df[
        "actual_failure"
    ] = y_test.values

    risk_df[
        "failure_probability_rank"
    ] = (
        risk_df[
            "failure_probability"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    risk_df[
        "risk_score_rank"
    ] = (
        risk_df[
            "unified_risk_score"
        ]
        .rank(
            ascending=False,
            method="min",
        )
    )

    # ---------------------------------------------------------
    # Save full risk table
    # ---------------------------------------------------------

    risk_output = (
        OUTPUT_DIR
        / "ai4i_unified_risk_scores.csv"
    )

    risk_df.to_csv(
        risk_output,
        index=False,
    )

    print(
        f"\nSaved risk table:"
    )

    print(
        f"  {risk_output}"
    )

    # ---------------------------------------------------------
    # Save high-risk subset
    # ---------------------------------------------------------

    high_risk = risk_df[
        risk_df[
            "unified_risk_score"
        ] >= 0.50
    ].copy()

    high_risk = high_risk.sort_values(
        "unified_risk_score",
        ascending=False,
    )

    high_risk_output = (
        OUTPUT_DIR
        / "high_risk_observations.csv"
    )

    high_risk.to_csv(
        high_risk_output,
        index=False,
    )

    print(
        f"\nHigh-risk observations "
        f"(score >= 0.50): "
        f"{len(high_risk)}"
    )

    print(
        f"Saved:"
    )

    print(
        f"  {high_risk_output}"
    )

    # ---------------------------------------------------------
    # Display highest-risk observations
    # ---------------------------------------------------------

    print(
        "\nTop 10 highest-risk observations:"
    )

    display_columns = [
        "failure_probability",
        "anomaly_score",
        "unified_risk_score",
        "risk_band",
        "actual_failure",
    ]

    print(
        risk_df.sort_values(
            "unified_risk_score",
            ascending=False,
        )[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print_summary(
        risk_df
    )

    # ---------------------------------------------------------
    # Figures
    # ---------------------------------------------------------

    print(
        "\nGenerating risk visualizations..."
    )

    create_risk_distribution(
        risk_df
    )

    create_risk_band_distribution(
        risk_df
    )

    create_risk_vs_failure_plot(
        risk_df
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "failure_probability_weight":
            FAILURE_WEIGHT,
        "anomaly_score_weight":
            ANOMALY_WEIGHT,
        "risk_bands": {
            "Low": "< 0.25",
            "Medium": ">= 0.25 and < 0.50",
            "High": ">= 0.50 and < 0.75",
            "Critical": ">= 0.75",
        },
        "supervised_model":
            "Gradient Boosting",
        "anomaly_model":
            "Isolation Forest",
        "dataset":
            "AI4I 2020 Predictive Maintenance Dataset",
        "random_state":
            RANDOM_STATE,
    }

    joblib.dump(
        metadata,
        OUTPUT_DIR
        / "risk_scoring_metadata.joblib",
    )

    # ---------------------------------------------------------
    # Complete
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "UNIFIED RISK SCORING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nForgeShield risk pipeline:"
    )

    print(
        "  Failure Probability"
    )

    print(
        "          +"
    )

    print(
        "    Anomaly Score"
    )

    print(
        "          ↓"
    )

    print(
        "  Unified Risk Score"
    )

    print(
        "          ↓"
    )

    print(
        "   Low / Medium / High / Critical"
    )

    print(
        "\nArtifacts:"
    )

    print(
        "  ✓ ai4i_unified_risk_scores.csv"
    )

    print(
        "  ✓ high_risk_observations.csv"
    )

    print(
        "  ✓ risk_scoring_metadata.joblib"
    )

    print(
        "  ✓ unified_risk_distribution.png"
    )

    print(
        "  ✓ risk_band_distribution.png"
    )

    print(
        "  ✓ risk_vs_failure.png"
    )


if __name__ == "__main__":
    main()

