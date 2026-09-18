from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ai4i"
MODEL_DIR = PROJECT_ROOT / "models" / "supervised"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42


def load_data():

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    return X_train, X_test, y_train, y_test


def calculate_scale_pos_weight(y):

    negative = (y == 0).sum()
    positive = (y == 1).sum()

    return negative / positive


def build_models(y_train):

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    models = {

        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_depth=8,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=7,
        ),

        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),

        "Naive Bayes": GaussianNB(),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    return models


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
    ).ravel()

    results = {

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

        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }

    return results


def train_models():

    print("\n" + "=" * 70)
    print("FORGESHIELD | SUPERVISED FAILURE PREDICTION")
    print("=" * 70)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_train, X_test, y_train, y_test = load_data()

    print("\nDataset:")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")
    print(f"  Features         : {X_train.shape[1]}")

    print("\nFailure distribution:")

    print(
        f"  Training failures: {int(y_train.sum())} "
        f"({y_train.mean() * 100:.2f}%)"
    )

    print(
        f"  Testing failures : {int(y_test.sum())} "
        f"({y_test.mean() * 100:.2f}%)"
    )

    models = build_models(y_train)

    all_results = []

    trained_models = {}

    print("\n" + "-" * 70)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_time = time.perf_counter() - start_time

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        metrics["model"] = name
        metrics["training_time_seconds"] = training_time

        all_results.append(metrics)

        trained_models[name] = model

        print(
            f"  Recall      : {metrics['recall']:.4f}"
        )

        print(
            f"  Precision   : {metrics['precision']:.4f}"
        )

        print(
            f"  F1          : {metrics['f1']:.4f}"
        )

        print(
            f"  ROC-AUC     : {metrics['roc_auc']:.4f}"
        )

        print(
            f"  PR-AUC      : {metrics['pr_auc']:.4f}"
        )

        print(
            f"  False Neg.  : {metrics['false_negatives']}"
        )

        print(
            f"  Train time  : {training_time:.3f}s"
        )

        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        joblib.dump(
            model,
            MODEL_DIR / f"{safe_name}.joblib",
        )

    results_df = pd.DataFrame(all_results)

    results_df = results_df[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "false_negatives",
            "false_positives",
            "true_positives",
            "true_negatives",
            "training_time_seconds",
        ]
    ]

    results_df = results_df.sort_values(
        by="recall",
        ascending=False,
    )

    results_path = REPORT_DIR / "supervised_model_results.csv"

    results_df.to_csv(
        results_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "false_negatives",
            ]
        ].to_string(index=False)
    )

    best_model_name = results_df.iloc[0]["model"]

    print("\n" + "=" * 70)
    print("CURRENT RECALL-LEADING MODEL")
    print("=" * 70)

    print(best_model_name)

    print(
        "\nNote: this is only a preliminary selection "
        "based on recall."
    )

    print(
        "\nResults saved to:"
        f"\n  {results_path}"
    )

    print(
        "\nModels saved to:"
        f"\n  {MODEL_DIR}"
    )

    print("\n" + "=" * 70)
    print("SUPERVISED MODELING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    train_models()
