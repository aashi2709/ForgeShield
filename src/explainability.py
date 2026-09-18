from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "explainability"
)

RANDOM_STATE = 42

# Keep SHAP computation manageable on CPU.
SHAP_SAMPLE_SIZE = 1000


def find_model(filename_candidates):
    """
    Find a model file from several possible filenames.
    This makes the script robust to the naming used during
    the supervised-model training stage.
    """

    for filename in filename_candidates:

        path = MODEL_DIR / filename

        if path.exists():
            return path

    # Fallback: search recursively
    for filename in filename_candidates:

        matches = list(
            MODEL_DIR.rglob(filename)
        )

        if matches:
            return matches[0]

    return None


def load_data():

    X_train = pd.read_csv(
        DATA_DIR / "X_train.csv"
    )

    X_test = pd.read_csv(
        DATA_DIR / "X_test.csv"
    )

    y_train = pd.read_csv(
        DATA_DIR / "y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        DATA_DIR / "y_test.csv"
    ).squeeze()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def calculate_shap_values(
    model,
    X_sample,
):

    print(
        "  Creating SHAP TreeExplainer..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    print(
        "  Calculating SHAP values..."
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    # Binary classification models can
    # return a list:
    #
    # [class_0_values, class_1_values]
    #
    # We want explanations for the
    # failure class.

    if isinstance(
        shap_values,
        list,
    ):

        if len(shap_values) == 2:

            shap_values = shap_values[1]

        else:

            shap_values = shap_values[-1]

    # Newer SHAP versions may return
    # an Explanation object or an array
    # with an output dimension.

    if hasattr(
        shap_values,
        "values",
    ):

        shap_values = (
            shap_values.values
        )

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 3:

        # Binary output dimension
        # usually sits on the final axis.

        if shap_values.shape[-1] == 2:

            shap_values = (
                shap_values[:, :, 1]
            )

        else:

            shap_values = (
                shap_values[:, :, 0]
            )

    return (
        explainer,
        shap_values,
    )


def save_feature_importance(
    shap_values,
    X_sample,
    model_name,
):

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    importance = pd.DataFrame(
        {
            "feature":
                X_sample.columns,
            "mean_abs_shap":
                mean_abs_shap,
        }
    )

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    output_path = (
        REPORT_DIR
        / f"{model_name}_shap_importance.csv"
    )

    importance.to_csv(
        output_path,
        index=False,
    )

    print(
        f"  Saved: {output_path.name}"
    )

    return importance


def create_summary_plot(
    shap_values,
    X_sample,
    model_name,
):

    plt.figure(
        figsize=(10, 7)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
    )

    plt.title(
        f"{model_name} SHAP Feature Importance"
    )

    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / f"{model_name}_shap_summary.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"  Saved: {output_path.name}"
    )


def create_bar_plot(
    shap_values,
    X_sample,
    model_name,
):

    plt.figure(
        figsize=(10, 7)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
    )

    plt.title(
        f"{model_name} Mean Absolute SHAP Values"
    )

    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / f"{model_name}_shap_bar.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"  Saved: {output_path.name}"
    )


def create_dependence_plots(
    shap_values,
    X_sample,
    importance,
    model_name,
    top_n=3,
):

    top_features = (
        importance
        .head(top_n)
        ["feature"]
        .tolist()
    )

    for feature in top_features:

        feature_index = (
            X_sample.columns
            .get_loc(feature)
        )

        plt.figure(
            figsize=(9, 6)
        )

        shap.dependence_plot(
            feature_index,
            shap_values,
            X_sample,
            show=False,
        )

        plt.title(
            f"{model_name}: SHAP Dependence - {feature}"
        )

        plt.tight_layout()

        safe_feature = (
            feature
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )

        output_path = (
            REPORT_DIR
            / (
                f"{model_name}_"
                f"dependence_{safe_feature}.png"
            )
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"  Saved: {output_path.name}"
        )


def create_global_comparison(
    all_importances,
):

    rows = []

    for model_name, importance in (
        all_importances.items()
    ):

        temp = importance.copy()

        temp["model"] = model_name

        rows.append(
            temp
        )

    combined = pd.concat(
        rows,
        ignore_index=True,
    )

    comparison = (
        combined
        .groupby("feature")[
            "mean_abs_shap"
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    comparison = comparison.sort_values(
        "mean",
        ascending=False,
    )

    comparison.to_csv(
        REPORT_DIR
        / "global_shap_comparison.csv",
        index=False,
    )

    print(
        "\nGlobal feature importance:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    return comparison


def main():

    print("\n" + "=" * 70)
    print(
        "FORGESHIELD | EXPLAINABLE AI WITH SHAP"
    )
    print("=" * 70)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nLoading processed AI4I data..."
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_data()

    print(
        f"  X_train: {X_train.shape}"
    )

    print(
        f"  X_test : {X_test.shape}"
    )

    print(
        f"  Features: {list(X_train.columns)}"
    )

    # ---------------------------------------------------------
    # Sample test observations
    # ---------------------------------------------------------

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X_test),
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    print(
        f"\nUsing {sample_size} test observations "
        f"for SHAP analysis."
    )

    # ---------------------------------------------------------
    # Locate models
    # ---------------------------------------------------------

    model_candidates = {
        "random_forest": [
            "random_forest.joblib",
            "random_forest.pkl",
            "Random Forest.joblib",
            "Random_Forest.joblib",
        ],
        "gradient_boosting": [
            "gradient_boosting.joblib",
            "gradient_boosting.pkl",
            "Gradient Boosting.joblib",
            "Gradient_Boosting.joblib",
        ],
        "xgboost": [
            "xgboost.joblib",
            "xgboost.pkl",
            "XGBoost.joblib",
            "XGB.joblib",
        ],
    }

    all_importances = {}

    # ---------------------------------------------------------
    # SHAP for tree-based models
    # ---------------------------------------------------------

    for model_name, candidates in (
        model_candidates.items()
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"MODEL: {model_name.upper()}"
        )

        model_path = find_model(
            candidates
        )

        if model_path is None:

            print(
                "  Model file not found."
            )

            print(
                "  Skipping this model."
            )

            continue

        print(
            f"  Loading: {model_path}"
        )

        model = joblib.load(
            model_path
        )

        explainer, shap_values = (
            calculate_shap_values(
                model,
                X_sample,
            )
        )

        print(
            f"  SHAP matrix: "
            f"{shap_values.shape}"
        )

        importance = (
            save_feature_importance(
                shap_values,
                X_sample,
                model_name,
            )
        )

        all_importances[
            model_name
        ] = importance

        print(
            "\n  Top features:"
        )

        print(
            importance.head(10).to_string(
                index=False
            )
        )

        create_summary_plot(
            shap_values,
            X_sample,
            model_name,
        )

        create_bar_plot(
            shap_values,
            X_sample,
            model_name,
        )

        create_dependence_plots(
            shap_values,
            X_sample,
            importance,
            model_name,
            top_n=3,
        )

        # -----------------------------------------------------
        # Save raw SHAP values for later integration
        # -----------------------------------------------------

        np.save(
            REPORT_DIR
            / f"{model_name}_shap_values.npy",
            shap_values,
        )

        X_sample.to_csv(
            REPORT_DIR
            / f"{model_name}_shap_sample.csv",
            index=False,
        )

    # ---------------------------------------------------------
    # Global comparison
    # ---------------------------------------------------------

    if all_importances:

        print(
            "\n" + "-" * 70
        )

        print(
            "CROSS-MODEL SHAP COMPARISON"
        )

        create_global_comparison(
            all_importances
        )

    else:

        print(
            "\nNo compatible tree models "
            "were found."
        )

        print(
            "Check the contents of:"
        )

        print(
            f"  {MODEL_DIR}"
        )

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EXPLAINABILITY ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nArtifacts are in:"
    )

    print(
        f"  {REPORT_DIR}"
    )

    print(
        "\nThese SHAP outputs will later feed "
        "the evidence layer of the ForgeShield "
        "incident intelligence pipeline."
    )


if __name__ == "__main__":
    main()
