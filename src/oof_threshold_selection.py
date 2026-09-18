from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
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
        "SVM": SVC(
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

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


def find_best_threshold(y_true, probabilities):

    thresholds = np.arange(
        0.05,
        0.96,
        0.01,
    )

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0,
        )

        false_negatives = (
            (y_true == 1) &
            (predictions == 0)
        ).sum()

        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_negatives": false_negatives,
            }
        )

    results = pd.DataFrame(rows)

    best = results.loc[
        results["f1"].idxmax()
    ]

    high_recall = results[
        results["recall"] >= 0.90
    ]

    return results, best, high_recall


def run_oof_threshold_selection():

    print("\n" + "=" * 70)
    print("FORGESHIELD | OOF THRESHOLD SELECTION")
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

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "threshold_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_summary = []

    for name, model in models.items():

        print("\n" + "-" * 70)
        print(f"Processing {name}...")

        oof_probabilities = np.zeros(
            len(y),
            dtype=float,
        )

        for fold, (train_idx, valid_idx) in enumerate(
            cv.split(X, y),
            start=1,
        ):

            fold_model = clone(model)

            fold_model.fit(
                X.iloc[train_idx],
                y.iloc[train_idx],
            )

            probabilities = fold_model.predict_proba(
                X.iloc[valid_idx]
            )[:, 1]

            oof_probabilities[
                valid_idx
            ] = probabilities

            print(
                f"  Fold {fold}/5 complete"
            )

        threshold_results, best, high_recall = (
            find_best_threshold(
                y,
                oof_probabilities,
            )
        )

        output_file = (
            output_dir
            / f"{name.lower().replace(' ', '_')}_oof_thresholds.csv"
        )

        threshold_results.to_csv(
            output_file,
            index=False,
        )

        print(
            f"\n  Best F1 threshold : "
            f"{best['threshold']:.2f}"
        )

        print(
            f"  Best F1           : "
            f"{best['f1']:.4f}"
        )

        if not high_recall.empty:

            highest_recall_threshold = (
                high_recall.iloc[-1]
            )

            print(
                "\n  Highest threshold "
                "with >=90% recall:"
            )

            print(
                f"    Threshold       : "
                f"{highest_recall_threshold['threshold']:.2f}"
            )

            print(
                f"    Precision       : "
                f"{highest_recall_threshold['precision']:.4f}"
            )

            print(
                f"    Recall          : "
                f"{highest_recall_threshold['recall']:.4f}"
            )

            print(
                f"    False negatives : "
                f"{int(highest_recall_threshold['false_negatives'])}"
            )

        threshold_summary.append(
            {
                "model": name,
                "best_f1_threshold": best["threshold"],
                "best_f1": best["f1"],
                "high_recall_threshold": (
                    highest_recall_threshold["threshold"]
                    if not high_recall.empty
                    else np.nan
                ),
                "high_recall_precision": (
                    highest_recall_threshold["precision"]
                    if not high_recall.empty
                    else np.nan
                ),
                "high_recall_recall": (
                    highest_recall_threshold["recall"]
                    if not high_recall.empty
                    else np.nan
                ),
                "high_recall_false_negatives": (
                    highest_recall_threshold[
                        "false_negatives"
                    ]
                    if not high_recall.empty
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(
        threshold_summary
    )

    summary = summary.sort_values(
        by="best_f1",
        ascending=False,
    )

    summary_path = (
        output_dir
        / "oof_threshold_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("OOF THRESHOLD SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(index=False)
    )

    print("\nResults saved to:")
    print(f"  {summary_path}")

    print("\n" + "=" * 70)
    print("OOF THRESHOLD SELECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_oof_threshold_selection()
