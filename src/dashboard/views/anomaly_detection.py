from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.components.theme import MUTED, RISK_COLORS
from dashboard.components.ui import empty_state, render_html


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ANOMALY_DIR = PROJECT_ROOT / "reports" / "figures" / "anomaly"


def _load_csv(filename: str) -> pd.DataFrame:
    path = ANOMALY_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)

        if pd.notna(number):
            return number

    except (TypeError, ValueError):
        pass

    return default


def _metric_card(label: str, value: str, detail: str) -> str:
    return f"""
    <div class="fs-detail-card">
        <div class="fs-detail-label">{label}</div>
        <div class="fs-detail-value">{value}</div>
        <div class="fs-detail-small">{detail}</div>
    </div>
    """


def render_anomaly_detection() -> None:
    render_html(
        """
        <div class="fs-eyebrow">ANOMALY DETECTION</div>
        <div class="fs-page-title">Unsupervised Risk Discovery</div>
        <div class="fs-page-subtitle">
            Detect abnormal machine behavior using isolation-based,
            boundary-based, and clustering methods.
        </div>
        """
    )

    st.write("")

    anomaly_df = _load_csv("anomaly_detector_comparison.csv")
    clustering_df = _load_csv("clustering_comparison.csv")
    silhouette_df = _load_csv("kmeans_silhouette.csv")

    if anomaly_df.empty:
        render_html(
            empty_state(
                "Anomaly results unavailable",
                "The anomaly-detection experiment artifacts could not be loaded.",
            )
        )
        return

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">Detector Overview</div>
        """
    )

    isolation_row = anomaly_df[
        anomaly_df["model"] == "Isolation Forest"
    ]

    svm_row = anomaly_df[
        anomaly_df["model"] == "One-Class SVM"
    ]

    isolation_detection = (
        _safe_float(isolation_row["detection_rate"].iloc[0])
        if not isolation_row.empty
        else 0.0
    )

    isolation_recall = (
        _safe_float(isolation_row["recall"].iloc[0])
        if not isolation_row.empty
        else 0.0
    )

    svm_detection = (
        _safe_float(svm_row["detection_rate"].iloc[0])
        if not svm_row.empty
        else 0.0
    )

    selected_k = 3

    if not silhouette_df.empty:
        best_row = silhouette_df.loc[
            silhouette_df["silhouette"].idxmax()
        ]
        selected_k = int(best_row["k"])

    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        render_html(
            _metric_card(
                "Isolation Forest",
                f"{isolation_detection * 100:.1f}%",
                "Test windows flagged",
            )
        )

    with k2:
        render_html(
            _metric_card(
                "IF Recall",
                f"{isolation_recall:.3f}",
                "Known failures detected",
            )
        )

    with k3:
        render_html(
            _metric_card(
                "One-Class SVM",
                f"{svm_detection * 100:.1f}%",
                "Test windows flagged",
            )
        )

    with k4:
        render_html(
            _metric_card(
                "K-Means",
                f"k = {selected_k}",
                "Selected by silhouette score",
            )
        )

    st.write("")

    # ------------------------------------------------------------------
    # Detector comparison
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">Anomaly Detector Comparison</div>
        """
    )

    detector_html = (
        '<div class="fs-panel">'
        '<div class="fs-table-wrap">'
        '<table class="fs-table">'
    )

    detector_html += """
        <thead>
            <tr>
                <th>Detector</th>
                <th class="num">Detection Rate</th>
                <th class="num">Precision</th>
                <th class="num">Recall</th>
                <th class="num">F1</th>
                <th class="num">Anomalies</th>
            </tr>
        </thead>
        <tbody>
    """

    for _, row in anomaly_df.iterrows():
        detector_html += f"""
            <tr>
                <td class="machine">{row["model"]}</td>
                <td class="num">
                    {_safe_float(row["detection_rate"]) * 100:.1f}%
                </td>
                <td class="num">
                    {_safe_float(row["precision"]):.3f}
                </td>
                <td class="num">
                    {_safe_float(row["recall"]):.3f}
                </td>
                <td class="num">
                    {_safe_float(row["f1"]):.3f}
                </td>
                <td class="num">
                    {int(_safe_float(row["anomalies_detected"])):,}
                </td>
            </tr>
        """

    detector_html += "</tbody></table></div></div>"

    render_html(detector_html)

    render_html(
        """
        <div class="fs-insight">
            Isolation Forest provides the continuous anomaly signal used by
            ForgeShield's unified risk layer. Binary anomaly labels shown
            here are evaluation outputs rather than the final risk score.
        </div>
        """
    )

    # ------------------------------------------------------------------
    # Clustering comparison
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">Clustering Structure Analysis</div>
        """
    )

    if not clustering_df.empty:
        clustering_html = (
            '<div class="fs-panel">'
            '<div class="fs-table-wrap">'
            '<table class="fs-table">'
        )

        clustering_html += """
            <thead>
                <tr>
                    <th>Method</th>
                    <th class="num">Clusters</th>
                    <th class="num">Silhouette</th>
                    <th class="num">Adjusted Rand Index</th>
                </tr>
            </thead>
            <tbody>
        """

        for _, row in clustering_df.iterrows():
            silhouette = _safe_float(row["silhouette"])
            ari = _safe_float(row["ARI"])

            clustering_html += f"""
                <tr>
                    <td class="machine">{row["model"]}</td>
                    <td class="num">
                        {int(_safe_float(row["clusters"]))}
                    </td>
                    <td class="num">
                        {silhouette:.3f}
                    </td>
                    <td class="num">
                        {ari:.3f}
                    </td>
                </tr>
            """

        clustering_html += "</tbody></table></div></div>"

        render_html(clustering_html)

    # ------------------------------------------------------------------
    # K-Means silhouette analysis
    # ------------------------------------------------------------------

    if not silhouette_df.empty:
        render_html(
            """
            <div class="fs-section-title">K-Means Model Selection</div>
            """
        )

        silhouette_display = silhouette_df.copy()

        silhouette_display["k"] = silhouette_display["k"].astype(int)
        silhouette_display["silhouette"] = silhouette_display[
            "silhouette"
        ].astype(float)

        chart_df = silhouette_display[
            ["k", "silhouette"]
        ].copy()

        # Plotly is used here instead of st.bar_chart so the small
        # silhouette values around 0.16-0.21 are displayed faithfully.
        import plotly.graph_objects as go

        fig_kmeans = go.Figure()

        fig_kmeans.add_trace(
            go.Bar(
                x=chart_df["k"],
                y=chart_df["silhouette"],
                name="Silhouette",
                hovertemplate=(
                    "k = %{x}"
                    "<br>Silhouette = %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        fig_kmeans.update_layout(
            height=320,
            margin=dict(
                l=10,
                r=20,
                t=20,
                b=10,
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                color="#AEB5BF",
            ),
            showlegend=False,
            xaxis=dict(
                title="k",
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
            ),
            yaxis=dict(
                title="Silhouette",
                range=[
                    0,
                    max(
                        0.24,
                        float(chart_df["silhouette"].max()) * 1.15,
                    ),
                ],
                gridcolor="rgba(255,255,255,0.06)",
                zeroline=False,
            ),
        )

        st.plotly_chart(
            fig_kmeans,
            use_container_width=True,
            config={
                "displayModeBar": False,
            },
        )

        render_html(
            f"""
            <div class="fs-insight">
                K-Means selection is based on silhouette score measured on
                the training data. The selected configuration uses
                <strong>k = {selected_k}</strong>.
            </div>
            """
        )

    # ------------------------------------------------------------------
    # PCA evidence
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">Feature-Space Evidence</div>
        """
    )

    pca_path = ANOMALY_DIR / "pca_failure_visualization.png"
    isolation_path = ANOMALY_DIR / "isolation_forest_pca.png"

    if pca_path.exists():
        render_html(
            """
            <div class="fs-panel">
                <div class="fs-state">
                    <div class="fs-state-title">
                        AI4I Test Data: PCA Projection
                    </div>
                    <div class="fs-state-copy">
                        Two-dimensional projection of the held-out test
                        feature space with machine-failure labels.
                    </div>
                </div>
            </div>
            """
        )

        st.image(
            str(pca_path),
            use_container_width=True,
        )

    if isolation_path.exists():
        render_html(
            """
            <div class="fs-panel">
                <div class="fs-state">
                    <div class="fs-state-title">
                        Isolation Forest Anomaly Map
                    </div>
                    <div class="fs-state-copy">
                        PCA projection showing normal observations and
                        observations flagged as anomalies by Isolation Forest.
                    </div>
                </div>
            </div>
            """
        )

        st.image(
            str(isolation_path),
            use_container_width=True,
        )

    # ------------------------------------------------------------------
    # Research interpretation
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">Research Interpretation</div>

        <div class="fs-panel">
            <div class="fs-state">
                <div class="fs-state-title">
                    Unsupervised discovery is complementary to failure
                    classification
                </div>

                <div class="fs-state-copy">
                    The clustering experiments do not show strong alignment
                    with the known machine-failure labels. Near-zero
                    Adjusted Rand Index values indicate that the discovered
                    clusters should not be interpreted as direct failure
                    classes. The anomaly detectors instead provide an
                    independent signal of unusual machine behavior.
                </div>
            </div>
        </div>

        <div class="fs-insight">
            For the ForgeShield decision layer, the Isolation Forest
            decision function is converted into a normalized continuous
            anomaly score and fused with supervised failure probability.
            This preserves more information than a binary anomaly flag.
        </div>
        """
    )

    # ------------------------------------------------------------------
    # Methodology
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">Methodology</div>

        <div class="fs-panel">
            <div class="fs-state">
                <div class="fs-state-title">
                    Train-only fitting
                </div>

                <div class="fs-state-copy">
                    Isolation Forest, One-Class SVM, K-Means, DBSCAN,
                    Agglomerative Clustering, and PCA are fitted using
                    training data. Models that support prediction are
                    evaluated on held-out test observations.
                </div>
            </div>
        </div>

        <div class="fs-footer">
            <span>FORGESHIELD · ANOMALY DETECTION</span>
            <span class="fs-footer-mono">
                AI4I 2020 · Unsupervised Research Proof of Concept
            </span>
        </div>
        """
    )