from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ai4i"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised"
THRESHOLD_PATH = (
    PROJECT_ROOT
    / "reports"
    / "threshold_analysis"
    / "oof_threshold_summary.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "supervised"
)

MODELS = {
    "Gradient Boosting": "gradient_boosting.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
    "SVM": "svm.joblib",
}


def load_test_data():
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_test = pd.read_csv(
        DATA_DIR / "y_test.csv"
    ).squeeze()

    return X_test, y_test


def load_thresholds():
    thresholds = pd.read_csv(THRESHOLD_PATH)

    thresholds["model"] = thresholds["model"].astype(str)

    required = {
        "model",
        "best_f1_threshold",
    }

    missing = required.difference(thresholds.columns)

    if missing:
        raise ValueError(
            "OOF threshold summary is missing columns: "
            + ", ".join(sorted(missing))
        )

    return thresholds.set_index("model")["best_f1_threshold"].to_dict()


def evaluate_model(
    name,
    model,
    threshold,
    X_test,
    y_test,
):
    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
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
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def run_evaluation():
    print("\n" + "=" * 72)
    print("FORGESHIELD | FINAL UNTOUCHED TEST EVALUATION")
    print("=" * 72)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_test, y_test = load_test_data()
    thresholds = load_thresholds()

    print(f"\nTest samples: {len(X_test)}")
    print(f"Test failures: {int(y_test.sum())}")
    print(f"Test normal: {int((y_test == 0).sum())}")

    results = []

    for name, filename in MODELS.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model: {model_path}"
            )

        if name not in thresholds:
            raise ValueError(
                f"No OOF threshold found for {name}"
            )

        model = joblib.load(model_path)

        threshold = float(thresholds[name])

        result = evaluate_model(
            name,
            model,
            threshold,
            X_test,
            y_test,
        )

        results.append(result)

        print(f"\n{name}")
        print(f"  Threshold : {threshold:.3f}")
        print(f"  Precision : {result['precision']:.4f}")
        print(f"  Recall    : {result['recall']:.4f}")
        print(f"  F1        : {result['f1']:.4f}")
        print(f"  ROC-AUC   : {result['roc_auc']:.4f}")
        print(f"  PR-AUC    : {result['pr_auc']:.4f}")
        print(
            "  Confusion : "
            f"TN={result['true_negatives']} "
            f"FP={result['false_positives']} "
            f"FN={result['false_negatives']} "
            f"TP={result['true_positives']}"
        )

    results_df = pd.DataFrame(results)

    output_path = (
        OUTPUT_DIR
        / "final_test_evaluation.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\nSaved:")
    print(f"  {output_path}")

    print("\n" + "=" * 72)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    run_evaluation()
