from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ai4i"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised"
RESULTS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "supervised"
    / "final_test_evaluation.csv"
)
FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "supervised"
)


MODEL_FILES = {
    "Gradient Boosting": "gradient_boosting.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "SVM": "svm.joblib",
}


def load_data():
    X_test = pd.read_csv(
        DATA_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        DATA_DIR / "y_test.csv"
    ).squeeze()

    results = pd.read_csv(RESULTS_PATH)

    return X_test, y_test, results


def generate_confusion_matrices(
    X_test,
    y_test,
    results,
):
    for name, filename in MODEL_FILES.items():

        row = results.loc[
            results["model"] == name
        ]

        if row.empty:
            raise ValueError(
                f"Missing final result for {name}"
            )

        threshold = float(
            row.iloc[0]["threshold"]
        )

        model = joblib.load(
            MODEL_DIR / filename
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= threshold
        ).astype(int)

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Normal",
                "Failure",
            ],
        )

        display.plot(
            ax=ax,
            values_format="d",
        )

        ax.set_title(
            f"{name} | Final Test | Threshold={threshold:.2f}"
        )

        plt.tight_layout()

        safe_name = (
            name.lower()
            .replace(" ", "_")
        )

        output_path = (
            FIGURE_DIR
            / f"07_final_confusion_{safe_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
        )

        plt.close()

        print(f"Saved: {output_path}")


def generate_metric_comparison(results):

    metrics = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    x = np.arange(
        len(results["model"])
    )

    width = 0.15

    plt.figure(
        figsize=(14, 7)
    )

    for i, metric in enumerate(metrics):

        plt.bar(
            x + i * width,
            results[metric],
            width,
            label=metric.upper(),
        )

    plt.xticks(
        x + width * 2,
        results["model"],
        rotation=25,
        ha="right",
    )

    plt.ylabel("Score")
    plt.title(
        "Final Test Performance at OOF-Selected Thresholds"
    )

    plt.ylim(
        0,
        1.05,
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "08_final_test_metric_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(f"Saved: {output_path}")


def run():
    print("\n" + "=" * 72)
    print("FORGESHIELD | FINAL TEST FIGURES")
    print("=" * 72)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_test, y_test, results = load_data()

    print(
        f"\nUsing locked results from:\n  {RESULTS_PATH}"
    )

    generate_confusion_matrices(
        X_test,
        y_test,
        results,
    )

    generate_metric_comparison(
        results
    )

    print("\n" + "=" * 72)
    print("FINAL TEST FIGURES COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    run()
