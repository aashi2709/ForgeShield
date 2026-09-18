from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ai4i"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "supervised"

MODELS_TO_ANALYZE = [
    "Svm",
    "Gradient Boosting",
    "Xgboost",
    "Random Forest",
]


def load_test_data():

    X_test = pd.read_csv(
        DATA_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        DATA_DIR / "y_test.csv"
    ).squeeze()

    return X_test, y_test


def load_model(filename):

    return joblib.load(
        MODEL_DIR / filename
    )


def analyze_thresholds(
    model,
    X_test,
    y_test,
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    records = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
        ).ravel()

        records.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "false_negatives": fn,
                "false_positives": fp,
                "true_positives": tp,
                "true_negatives": tn,
            }
        )

    return pd.DataFrame(records)


def plot_precision_recall(
    model_results,
    model_name,
):

    plt.figure(figsize=(9, 6))

    plt.plot(
        model_results["threshold"],
        model_results["precision"],
        label="Precision",
        linewidth=2,
    )

    plt.plot(
        model_results["threshold"],
        model_results["recall"],
        label="Recall",
        linewidth=2,
    )

    plt.plot(
        model_results["threshold"],
        model_results["f1"],
        label="F1",
        linewidth=2,
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title(
        f"{model_name} | Threshold Performance"
    )

    plt.ylim(0, 1.05)

    plt.legend()

    plt.tight_layout()

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    plt.savefig(
        FIGURE_DIR
        / f"05_threshold_metrics_{safe_name}.png",
        dpi=300,
    )

    plt.close()


def plot_false_negatives(
    model_results,
    model_name,
):

    plt.figure(figsize=(9, 6))

    plt.plot(
        model_results["threshold"],
        model_results["false_negatives"],
        linewidth=2,
    )

    plt.xlabel("Decision Threshold")
    plt.ylabel("False Negatives")
    plt.title(
        f"{model_name} | Missed Failures vs Threshold"
    )

    plt.tight_layout()

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    plt.savefig(
        FIGURE_DIR
        / f"06_false_negatives_{safe_name}.png",
        dpi=300,
    )

    plt.close()


def run_threshold_analysis():

    print("\n" + "=" * 70)
    print("FORGESHIELD | DECISION THRESHOLD ANALYSIS")
    print("=" * 70)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_test, y_test = load_test_data()

    model_files = {
        "SVM": "svm.joblib",
        "Gradient Boosting": "gradient_boosting.joblib",
        "XGBoost": "xgboost.joblib",
        "Random Forest": "random_forest.joblib",
    }

    for model_name, filename in model_files.items():

        print(f"\nAnalyzing {model_name}...")

        model = load_model(filename)

        results = analyze_thresholds(
            model,
            X_test,
            y_test,
        )

        results.to_csv(
            FIGURE_DIR
            / (
                "threshold_results_"
                + model_name.lower().replace(" ", "_")
                + ".csv"
            ),
            index=False,
        )

        plot_precision_recall(
            results,
            model_name,
        )

        plot_false_negatives(
            results,
            model_name,
        )

        # Best F1 threshold
        best_f1 = results.loc[
            results["f1"].idxmax()
        ]

        # First threshold achieving at least 90% recall
        high_recall = results[
            results["recall"] >= 0.90
        ]

        print(
            f"  Best F1 threshold : "
            f"{best_f1['threshold']:.2f}"
        )

        print(
            f"  Best F1           : "
            f"{best_f1['f1']:.4f}"
        )

        if not high_recall.empty:

            first = high_recall.iloc[-1]

            print(
                f"  90%+ recall range : "
                f"{high_recall['threshold'].min():.2f}"
                f" - "
                f"{high_recall['threshold'].max():.2f}"
            )

            print(
                f"  At highest threshold "
                f"with >=90% recall:"
            )

            print(
                f"    Precision        : "
                f"{first['precision']:.4f}"
            )

            print(
                f"    Recall           : "
                f"{first['recall']:.4f}"
            )

            print(
                f"    False negatives  : "
                f"{int(first['false_negatives'])}"
            )

    print("\n" + "=" * 70)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_threshold_analysis()
