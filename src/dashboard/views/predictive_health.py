from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from dashboard.components.theme import MUTED, RISK_COLORS
from dashboard.components.ui import empty_state, render_html


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CMAPSS_DIR = PROJECT_ROOT / "reports" / "figures" / "cmapss"


def _load_csv(filename: str) -> pd.DataFrame:
    path = CMAPSS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(value, default=0.0) -> float:
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


def render_predictive_health() -> None:
    render_html(
        """
        <div class="fs-eyebrow">PREDICTIVE HEALTH</div>
        <div class="fs-page-title">Remaining Useful Life Analysis</div>
        <div class="fs-page-subtitle">
            Analyze degradation behavior using C-MAPSS run-to-failure
            trajectories and recurrent deep-learning models.
        </div>
        """
    )

    st.write("")

    model_df = _load_csv("cmapss_test_model_comparison.csv")
    engine_df = _load_csv("cmapss_per_engine_results.csv")
    error_df = _load_csv("cmapss_error_statistics.csv")
    autoencoder_df = _load_csv("lstm_autoencoder_test_results.csv")

    if model_df.empty:
        render_html(
            empty_state(
                "C-MAPSS results unavailable",
                "The predictive-health experiment artifacts could not be loaded.",
            )
        )
        return

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    test_windows = 10196
    near_failure_windows = 332
    engine_count = 100

    lstm_mae = _safe_float(
        model_df.loc[model_df["model"] == "LSTM", "MAE"].iloc[0]
    )
    cnn_lstm_mae = _safe_float(
        model_df.loc[model_df["model"] == "CNN-LSTM", "MAE"].iloc[0]
    )
    gru_mae = _safe_float(
        model_df.loc[model_df["model"] == "GRU", "MAE"].iloc[0]
    )

    k1, k2, k3, k4 = st.columns(4, gap="large")

    with k1:
        render_html(
            _metric_card(
                "Test Windows",
                f"{test_windows:,}",
                "Held-out C-MAPSS windows",
            )
        )

    with k2:
        render_html(
            _metric_card(
                "Test Engines",
                str(engine_count),
                "Engine-level evaluation",
            )
        )

    with k3:
        render_html(
            _metric_card(
                "Near-Failure Windows",
                f"{near_failure_windows:,}",
                "RUL ≤ 30 cycles",
            )
        )

    with k4:
        render_html(
            _metric_card(
                "LSTM MAE",
                f"{lstm_mae:.2f}",
                "RUL cycles",
            )
        )

    st.write("")

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">RUL Model Comparison</div>
        """
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=model_df["model"],
            y=model_df["MAE"],
            name="MAE",
            hovertemplate="%{x}<br>MAE: %{y:.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=model_df["model"],
            y=model_df["RMSE"],
            name="RMSE",
            hovertemplate="%{x}<br>RMSE: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        barmode="group",
        height=360,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#AEB5BF"),
        xaxis=dict(
            title=None,
            gridcolor="rgba(255,255,255,0.06)",
        ),
        yaxis=dict(
            title="RUL Error (cycles)",
            gridcolor="rgba(255,255,255,0.06)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # ------------------------------------------------------------------
    # Error statistics
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">Prediction Error Statistics</div>
        """
    )

    if not error_df.empty:
        error_html = '<div class="fs-panel"><div class="fs-table-wrap"><table class="fs-table">'
        error_html += """
        <thead>
            <tr>
                <th>Model</th>
                <th class="num">Mean Error</th>
                <th class="num">Median Absolute Error</th>
                <th class="num">Std. Error</th>
                <th class="num">Max Absolute Error</th>
            </tr>
        </thead>
        <tbody>
        """

        for _, row in error_df.iterrows():
            error_html += f"""
            <tr>
                <td class="machine">{row["model"]}</td>
                <td class="num">{_safe_float(row["mean_error"]):.2f}</td>
                <td class="num">{_safe_float(row["median_absolute_error"]):.2f}</td>
                <td class="num">{_safe_float(row["std_error"]):.2f}</td>
                <td class="num">{_safe_float(row["max_absolute_error"]):.2f}</td>
            </tr>
            """

        error_html += "</tbody></table></div></div>"
        render_html(error_html)

    # ------------------------------------------------------------------
    # Engine explorer
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">Engine-Level Analysis</div>
        """
    )

    if not engine_df.empty:
        selected_engine = st.selectbox(
            "Engine",
            engine_df["unit_id"].astype(int).tolist(),
            index=0,
            key="predictive_health_engine",
            label_visibility="collapsed",
        )

        engine_row = engine_df[
            engine_df["unit_id"].astype(int) == int(selected_engine)
        ].iloc[0]

        e1, e2, e3, e4 = st.columns(4, gap="large")

        with e1:
            render_html(
                _metric_card(
                    "Engine",
                    f"{int(engine_row['unit_id']):03d}",
                    "Held-out test engine",
                )
            )

        with e2:
            render_html(
                _metric_card(
                    "Windows",
                    f"{int(engine_row['num_windows']):,}",
                    "Evaluation windows",
                )
            )

        with e3:
            render_html(
                _metric_card(
                    "LSTM MAE",
                    f"{_safe_float(engine_row['LSTM_MAE']):.2f}",
                    "RUL cycles",
                )
            )

        with e4:
            render_html(
                _metric_card(
                    "CNN-LSTM MAE",
                    f"{_safe_float(engine_row['CNN-LSTM_MAE']):.2f}",
                    "RUL cycles",
                )
            )

        engine_models = ["LSTM_MAE", "GRU_MAE", "CNN-LSTM_MAE"]
        engine_values = [
            _safe_float(engine_row[column])
            for column in engine_models
        ]

        fig_engine = go.Figure(
            go.Bar(
                x=["LSTM", "GRU", "CNN-LSTM"],
                y=engine_values,
                hovertemplate="%{x}<br>MAE: %{y:.2f}<extra></extra>",
            )
        )

        fig_engine.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#AEB5BF"),
            yaxis=dict(
                title="MAE (cycles)",
                gridcolor="rgba(255,255,255,0.06)",
            ),
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.06)",
            ),
        )

        st.plotly_chart(
            fig_engine,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ------------------------------------------------------------------
    # Autoencoder
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">Reconstruction-Based Anomaly Evidence</div>
        """
    )

    if not autoencoder_df.empty:
        ae = autoencoder_df.iloc[0]

        precision = _safe_float(ae["precision"])
        recall = _safe_float(ae["recall"])
        f1 = _safe_float(ae["f1"])
        threshold = _safe_float(ae["threshold"])
        anomalies = int(_safe_float(ae["anomalies_detected"]))

        a1, a2, a3, a4 = st.columns(4, gap="large")

        with a1:
            render_html(
                _metric_card(
                    "Threshold",
                    f"{threshold:.3f}",
                    "Frozen from validation",
                )
            )

        with a2:
            render_html(
                _metric_card(
                    "Precision",
                    f"{precision:.3f}",
                    "Held-out test",
                )
            )

        with a3:
            render_html(
                _metric_card(
                    "Recall",
                    f"{recall:.3f}",
                    "Near-failure detection",
                )
            )

        with a4:
            render_html(
                _metric_card(
                    "F1",
                    f"{f1:.3f}",
                    f"{anomalies:,} anomalies detected",
                )
            )

        test_error_path = (
            PROJECT_ROOT
            / "models"
            / "cmapss"
            / "lstm_autoencoder_test_errors.npy"
        )

        import numpy as np

        test_errors = np.load(test_error_path)

        st.markdown(
            "#### Test Reconstruction Error Distribution"
        )

        hist_counts, bin_edges = np.histogram(
            test_errors,
            bins=45,
        )

        hist_df = pd.DataFrame(
            {
                "Reconstruction Error": bin_edges[:-1],
                "Test Windows": hist_counts,
            }
        )

        st.bar_chart(
            hist_df,
            x="Reconstruction Error",
            y="Test Windows",
            height=360,
        )

        render_html(
            """
            <div class="fs-insight">
                The LSTM autoencoder learns reconstruction behavior from
                healthy training windows. Its reconstruction-error threshold
                is selected on the engine-wise validation split and then
                applied unchanged to the held-out test set.
            </div>
            """
        )

    # ------------------------------------------------------------------
    # Methodology note
    # ------------------------------------------------------------------

    render_html(
        """
        <div class="fs-section-title">Research Context</div>

        <div class="fs-panel">
            <div class="fs-state">
                <div class="fs-state-title">Engine-wise evaluation</div>
                <div class="fs-state-copy">
                    C-MAPSS trajectories are split by engine unit rather than
                    by individual windows. This prevents windows from the same
                    physical engine trajectory appearing across training and
                    validation sets. RUL predictions are evaluated on the
                    held-out test engines.
                </div>
            </div>
        </div>

        <div class="fs-footer">
            <span>FORGESHIELD · PREDICTIVE HEALTH</span>
            <span class="fs-footer-mono">
                C-MAPSS FD001 · Research Proof of Concept
            </span>
        </div>
        """
    )
