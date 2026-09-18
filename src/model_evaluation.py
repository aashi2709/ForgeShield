from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    auc,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ai4i"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures" / "supervised"


def load_data():

    X_test = pd.read_csv(
        DATA_DIR / "X_test.csv"
    )

    y_test = pd.read_csv(
        DATA_DIR / "y_test.csv"
    ).squeeze()

    return X_test, y_test


def load_models():

    models = {}

    for path in MODEL_DIR.glob("*.joblib"):

        name = path.stem.replace("_", " ").title()

        models[name] = joblib.load(path)

    return models


def plot_roc_curves(models, X_test, y_test):

    plt.figure(figsize=(9, 7))

    for name, model in models.items():

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities,
        )

        roc_auc = auc(
            fpr,
            tpr,
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC={roc_auc:.3f})",
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Machine Failure Prediction")

    plt.legend(
        loc="lower right",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "01_roc_curves.png",
        dpi=300,
    )

    plt.close()


def plot_precision_recall_curves(
    models,
    X_test,
    y_test,
):

    plt.figure(figsize=(9, 7))

    for name, model in models.items():

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        precision, recall, _ = precision_recall_curve(
            y_test,
            probabilities,
        )

        pr_auc = auc(
            recall,
            precision,
        )

        plt.plot(
            recall,
            precision,
            linewidth=2,
            label=f"{name} (AUC={pr_auc:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        "Precision-Recall Curves for Machine Failure Prediction"
    )

    plt.legend(
        loc="lower left",
        fontsize=8,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "02_precision_recall_curves.png",
        dpi=300,
    )

    plt.close()


def plot_confusion_matrices(
    models,
    X_test,
    y_test,
):

    for name, model in models.items():

        predictions = model.predict(
            X_test
        )

        cm = confusion_matrix(
            y_test,
            predictions,
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
            f"{name} | Confusion Matrix"
        )

        plt.tight_layout()

        safe_name = (
            name.lower()
            .replace(" ", "_")
        )

        plt.savefig(
            FIGURE_DIR / f"03_confusion_{safe_name}.png",
            dpi=300,
        )

        plt.close()


def plot_metric_comparison():

    results_path = (
        PROJECT_ROOT
        / "reports"
        / "supervised_model_results.csv"
    )

    df = pd.read_csv(
        results_path
    )

    metrics = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    x = np.arange(
        len(df["model"])
    )

    width = 0.15

    plt.figure(
        figsize=(14, 7)
    )

    for i, metric in enumerate(metrics):

        plt.bar(
            x + i * width,
            df[metric],
            width,
            label=metric.upper(),
        )

    plt.xticks(
        x + width * 2,
        df["model"],
        rotation=35,
        ha="right",
    )

    plt.ylabel("Score")
    plt.title(
        "Supervised Model Performance Comparison"
    )

    plt.ylim(
        0,
        1.05,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "04_model_metric_comparison.png",
        dpi=300,
    )

    plt.close()


def run_evaluation():

    print("\n" + "=" * 70)
    print("FORGESHIELD | MODEL EVALUATION")
    print("=" * 70)

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_test, y_test = load_data()

    models = load_models()

    print(
        f"\nModels loaded: {len(models)}"
    )

    for name in models:
        print(f"  - {name}")

    print("\nGenerating ROC curves...")

    plot_roc_curves(
        models,
        X_test,
        y_test,
    )

    print("Generating Precision-Recall curves...")

    plot_precision_recall_curves(
        models,
        X_test,
        y_test,
    )

    print("Generating confusion matrices...")

    plot_confusion_matrices(
        models,
        X_test,
        y_test,
    )

    print("Generating metric comparison...")

    plot_metric_comparison()

    print("\nFigures saved to:")

    print(
        f"  {FIGURE_DIR}"
    )

    print("\n" + "=" * 70)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
