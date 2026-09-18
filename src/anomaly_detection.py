from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.svm import OneClassSVM


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ai4i"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "anomaly"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "anomaly"
)


RANDOM_STATE = 42


def load_data():

    X_train = pd.read_csv(
        PROCESSED_DIR / "X_train.csv"
    )

    X_test = pd.read_csv(
        PROCESSED_DIR / "X_test.csv"
    )

    y_train = pd.read_csv(
        PROCESSED_DIR / "y_train.csv"
    ).squeeze()

    y_test = pd.read_csv(
        PROCESSED_DIR / "y_test.csv"
    ).squeeze()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def evaluate_binary_detector(
    name,
    scores,
    y_true,
):
    """
    Convert anomaly scores into anomaly labels.

    For Isolation Forest:
        predict = -1 → anomaly
        predict = +1 → normal

    For One-Class SVM:
        predict = -1 → anomaly
        predict = +1 → normal
    """

    if name == "Isolation Forest":
        predictions = scores

    elif name == "One-Class SVM":
        predictions = scores

    else:
        raise ValueError(
            f"Unsupported detector: {name}"
        )

    anomaly_labels = (
        predictions == -1
    ).astype(int)

    true_labels = np.asarray(
        y_true
    ).astype(int)

    anomaly_count = anomaly_labels.sum()

    total = len(anomaly_labels)

    true_failures = true_labels.sum()

    true_positives = np.sum(
        (anomaly_labels == 1)
        & (true_labels == 1)
    )

    false_positives = np.sum(
        (anomaly_labels == 1)
        & (true_labels == 0)
    )

    false_negatives = np.sum(
        (anomaly_labels == 0)
        & (true_labels == 1)
    )

    precision = (
        true_positives
        / (true_positives + false_positives)
        if (
            true_positives
            + false_positives
        ) > 0
        else 0.0
    )

    recall = (
        true_positives
        / (true_positives + false_negatives)
        if (
            true_positives
            + false_negatives
        ) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    print(
        f"\n{name}"
    )

    print(
        f"  Detected anomalies : "
        f"{anomaly_count}"
    )

    print(
        f"  Detection rate     : "
        f"{anomaly_count / total:.4f}"
    )

    print(
        f"  True failures      : "
        f"{true_failures}"
    )

    print(
        f"  Precision          : "
        f"{precision:.4f}"
    )

    print(
        f"  Recall             : "
        f"{recall:.4f}"
    )

    print(
        f"  F1                 : "
        f"{f1:.4f}"
    )

    return {
        "model": name,
        "anomalies_detected": anomaly_count,
        "detection_rate": anomaly_count / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_clustering(
    name,
    labels,
    X,
    y_true,
):

    unique_labels = np.unique(
        labels
    )

    non_noise_labels = [
        label
        for label in unique_labels
        if label != -1
    ]

    cluster_count = len(
        non_noise_labels
    )

    if cluster_count < 2:

        silhouette = np.nan

    else:

        mask = labels != -1

        if (
            mask.sum() > 1
            and len(
                np.unique(
                    labels[mask]
                )
            ) >= 2
        ):

            silhouette = (
                silhouette_score(
                    X[mask],
                    labels[mask],
                )
            )

        else:

            silhouette = np.nan

    ari = adjusted_rand_score(
        y_true,
        labels,
    )

    print(
        f"\n{name}"
    )

    print(
        f"  Clusters           : "
        f"{cluster_count}"
    )

    print(
        f"  Silhouette Score   : "
        f"{silhouette:.4f}"
    )

    print(
        f"  Adjusted Rand Index: "
        f"{ari:.4f}"
    )

    return {
        "model": name,
        "clusters": cluster_count,
        "silhouette": silhouette,
        "ARI": ari,
    }


def main():

    print("\n" + "=" * 70)
    print(
        "FORGESHIELD | UNSUPERVISED ANOMALY DETECTION"
    )
    print("=" * 70)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load AI4I data
    # ---------------------------------------------------------

    print(
        "\nLoading AI4I processed data..."
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
        f"  Failures in test: "
        f"{y_test.sum()}"
    )

    # ---------------------------------------------------------
    # Isolation Forest
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "ISOLATION FOREST"
    )

    isolation_forest = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    isolation_forest.fit(
        X_train
    )

    isolation_predictions = (
        isolation_forest.predict(
            X_test
        )
    )

    isolation_results = (
        evaluate_binary_detector(
            "Isolation Forest",
            isolation_predictions,
            y_test,
        )
    )

    joblib.dump(
        isolation_forest,
        MODEL_DIR
        / "isolation_forest.joblib",
    )

    # ---------------------------------------------------------
    # One-Class SVM
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "ONE-CLASS SVM"
    )

    one_class_svm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.05,
    )

    one_class_svm.fit(
        X_train
    )

    svm_predictions = (
        one_class_svm.predict(
            X_test
        )
    )

    svm_results = (
        evaluate_binary_detector(
            "One-Class SVM",
            svm_predictions,
            y_test,
        )
    )

    joblib.dump(
        one_class_svm,
        MODEL_DIR
        / "one_class_svm.joblib",
    )

    # ---------------------------------------------------------
    # K-Means
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "K-MEANS CLUSTERING"
    )

    kmeans_scores = []

    for k in range(2, 7):

        kmeans = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )

        labels = kmeans.fit_predict(
            X_test
        )

        score = silhouette_score(
            X_test,
            labels,
        )

        kmeans_scores.append(
            {
                "k": k,
                "silhouette": score,
            }
        )

        print(
            f"  k={k} | "
            f"Silhouette={score:.4f}"
        )

    kmeans_scores_df = pd.DataFrame(
        kmeans_scores
    )

    best_k = int(
        kmeans_scores_df.loc[
            kmeans_scores_df[
                "silhouette"
            ].idxmax(),
            "k",
        ]
    )

    print(
        f"\nSelected K-Means k: {best_k}"
    )

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10,
    )

    kmeans_labels = (
        kmeans.fit_predict(
            X_test
        )
    )

    kmeans_results = evaluate_clustering(
        "K-Means",
        kmeans_labels,
        X_test,
        y_test,
    )

    kmeans_scores_df.to_csv(
        REPORT_DIR
        / "kmeans_silhouette.csv",
        index=False,
    )

    joblib.dump(
        kmeans,
        MODEL_DIR
        / "kmeans.joblib",
    )

    # ---------------------------------------------------------
    # DBSCAN
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "DBSCAN"
    )

    dbscan = DBSCAN(
        eps=0.8,
        min_samples=10,
    )

    dbscan_labels = (
        dbscan.fit_predict(
            X_test
        )
    )

    dbscan_results = evaluate_clustering(
        "DBSCAN",
        dbscan_labels,
        X_test,
        y_test,
    )

    joblib.dump(
        dbscan,
        MODEL_DIR
        / "dbscan.joblib",
    )

    # ---------------------------------------------------------
    # Agglomerative clustering
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "AGGLOMERATIVE CLUSTERING"
    )

    agglomerative = (
        AgglomerativeClustering(
            n_clusters=3,
            linkage="ward",
        )
    )

    agglomerative_labels = (
        agglomerative.fit_predict(
            X_test
        )
    )

    agglomerative_results = (
        evaluate_clustering(
            "Agglomerative",
            agglomerative_labels,
            X_test,
            y_test,
        )
    )

    joblib.dump(
        agglomerative,
        MODEL_DIR
        / "agglomerative.joblib",
    )

    # ---------------------------------------------------------
    # Save clustering results
    # ---------------------------------------------------------

    clustering_results = pd.DataFrame(
        [
            kmeans_results,
            dbscan_results,
            agglomerative_results,
        ]
    )

    clustering_results.to_csv(
        REPORT_DIR
        / "clustering_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Save anomaly comparison
    # ---------------------------------------------------------

    anomaly_results = pd.DataFrame(
        [
            isolation_results,
            svm_results,
        ]
    )

    anomaly_results.to_csv(
        REPORT_DIR
        / "anomaly_detector_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # PCA visualization
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "GENERATING PCA VISUALIZATION"
    )

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    X_pca = pca.fit_transform(
        X_test
    )

    plt.figure(
        figsize=(10, 7)
    )

    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_test,
        alpha=0.6,
        s=15,
    )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.title(
        "AI4I Test Data: PCA Projection"
    )

    plt.colorbar(
        scatter,
        label="Machine Failure"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "pca_failure_visualization.png",
        dpi=300,
    )

    plt.close()

    # ---------------------------------------------------------
    # Isolation Forest visualization
    # ---------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        X_pca[
            isolation_predictions == 1,
            0
        ],
        X_pca[
            isolation_predictions == 1,
            1
        ],
        alpha=0.35,
        s=12,
        label="Normal",
    )

    plt.scatter(
        X_pca[
            isolation_predictions == -1,
            0
        ],
        X_pca[
            isolation_predictions == -1,
            1
        ],
        alpha=0.8,
        s=20,
        label="Detected Anomaly",
    )

    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.title(
        "Isolation Forest Anomaly Detection"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "isolation_forest_pca.png",
        dpi=300,
    )

    plt.close()

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ANOMALY DETECTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved results:"
    )

    print(
        "  ✓ anomaly_detector_comparison.csv"
    )

    print(
        "  ✓ clustering_comparison.csv"
    )

    print(
        "  ✓ kmeans_silhouette.csv"
    )

    print(
        "  ✓ pca_failure_visualization.png"
    )

    print(
        "  ✓ isolation_forest_pca.png"
    )

    print(
        f"\nFigures directory:"
    )

    print(
        f"  {REPORT_DIR}"
    )


if __name__ == "__main__":
    main()
