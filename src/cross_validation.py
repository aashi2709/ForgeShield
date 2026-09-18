from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "ai4i"

X_PATH = DATA_DIR / "X_train.csv"
Y_PATH = DATA_DIR / "y_train.csv"

RANDOM_STATE = 42
N_SPLITS = 5


def load_data():
    X = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).squeeze("columns")
    return X, y


def build_models(y):
    negative = (y == 0).sum()
    positive = (y == 1).sum()

    scale_pos_weight = negative / positive

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=8,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    class_weight="balanced",
                    probability=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "Gradient Boosting": GradientBoostingClassifier(
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


def run_cross_validation():

    print("\n" + "=" * 70)
    print("FORGESHIELD | STRATIFIED CROSS-VALIDATION")
    print("=" * 70)

    X, y = load_data()

    print(f"\nTraining samples : {len(X)}")
    print(f"Features         : {X.shape[1]}")
    print(f"Failure samples  : {y.sum()}")
    print(f"Failure rate     : {y.mean() * 100:.2f}%")

    models = build_models(y)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "precision": make_scorer(
            precision_score,
            zero_division=0,
        ),
        "recall": make_scorer(
            recall_score,
            zero_division=0,
        ),
        "f1": make_scorer(
            f1_score,
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results = []

    print("\n" + "-" * 70)

    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        row = {
            "model": name,
            "precision_mean": scores["test_precision"].mean(),
            "precision_std": scores["test_precision"].std(),
            "recall_mean": scores["test_recall"].mean(),
            "recall_std": scores["test_recall"].std(),
            "f1_mean": scores["test_f1"].mean(),
            "f1_std": scores["test_f1"].std(),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_std": scores["test_roc_auc"].std(),
            "pr_auc_mean": scores["test_pr_auc"].mean(),
            "pr_auc_std": scores["test_pr_auc"].std(),
        }

        results.append(row)

        print(
            f"  Recall : {row['recall_mean']:.4f} "
            f"± {row['recall_std']:.4f}"
        )

        print(
            f"  F1     : {row['f1_mean']:.4f} "
            f"± {row['f1_std']:.4f}"
        )

        print(
            f"  ROC-AUC: {row['roc_auc_mean']:.4f} "
            f"± {row['roc_auc_std']:.4f}"
        )

        print(
            f"  PR-AUC : {row['pr_auc_mean']:.4f} "
            f"± {row['pr_auc_std']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="recall_mean",
        ascending=False,
    )

    output_dir = PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "cross_validation_results.csv"

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print(
        results_df[
            [
                "model",
                "recall_mean",
                "f1_mean",
                "roc_auc_mean",
                "pr_auc_mean",
            ]
        ].to_string(index=False)
    )

    print("\nResults saved to:")
    print(f"  {output_path}")

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_cross_validation()
