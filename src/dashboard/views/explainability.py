from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.theme import MUTED, RISK_COLORS
from dashboard.components.ui import empty_state, render_html


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPLAINABILITY_DIR = PROJECT_ROOT / "reports" / "figures" / "explainability"


def _load_csv(filename: str) -> pd.DataFrame:
    """Load an explainability CSV artifact safely."""
    path = EXPLAINABILITY_DIR / filename

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _image_path(filename: str) -> str | None:
    """Return an existing explainability image path."""
    path = EXPLAINABILITY_DIR / filename

    if path.exists():
        return str(path)

    return None


def _metric_card(label: str, value: str, detail: str = "") -> str:
    """Render a ForgeShield-style metric card."""
    return f"""
    <div class="fs-card" style="padding:18px 20px; min-height:108px;">
        <div style="
            font-size:11px;
            letter-spacing:0.12em;
            text-transform:uppercase;
            color:{MUTED};
            font-weight:700;
            margin-bottom:10px;
        ">
            {label}
        </div>

        <div style="
            font-size:27px;
            font-weight:750;
            color:#F5F7FA;
            line-height:1.1;
        ">
            {value}
        </div>

        <div style="
            font-size:12px;
            color:{MUTED};
            margin-top:7px;
        ">
            {detail}
        </div>
    </div>
    """


def render_explainability() -> None:
    """Render the ForgeShield Explainability research view."""

    st.markdown(
        """
        <div class="fs-page-kicker">EXPLAINABILITY</div>

        <div class="fs-page-title">
            Model Decision Evidence
        </div>

        <div class="fs-page-subtitle">
            SHAP-based evidence showing which features influence failure predictions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    global_df = _load_csv("global_shap_comparison.csv")
    gradient_df = _load_csv("gradient_boosting_shap_importance.csv")
    random_forest_df = _load_csv("random_forest_shap_importance.csv")
    xgboost_df = _load_csv("xgboost_shap_importance.csv")

    model_frames = {
        "Gradient Boosting": gradient_df,
        "Random Forest": random_forest_df,
        "XGBoost": xgboost_df,
    }

    available_models = [
        name for name, frame in model_frames.items()
        if not frame.empty
    ]

    if global_df.empty and not available_models:
        empty_state(
            "Explainability artifacts are not available.",
            "Run the explainability pipeline to generate SHAP evidence.",
        )
        return

    # ------------------------------------------------------------------
    # Overview
    # ------------------------------------------------------------------

    feature_count = len(global_df) if not global_df.empty else 0

    top_feature = (
        str(global_df.iloc[0]["feature"])
        if not global_df.empty and "feature" in global_df.columns
        else "Available"
    )

    model_count = len(available_models)

    # IMPORTANT: use render_html() here. The dashboard's HTML helper uses
    # st.html(), preventing the metric-card markup from being displayed
    # literally as text.
    render_html(
        f"""
        <div style="
            display:grid;
            grid-template-columns:repeat(3, 1fr);
            gap:14px;
            margin:18px 0 26px 0;
        ">
            {_metric_card(
                "Models explained",
                str(model_count),
                "Tree-based failure classifiers"
            )}

            {_metric_card(
                "Features analyzed",
                str(feature_count),
                "Global SHAP feature evidence"
            )}

            {_metric_card(
                "Highest global influence",
                top_feature,
                "Mean absolute SHAP comparison"
            )}
        </div>
        """
    )

    # ------------------------------------------------------------------
    # Global SHAP importance
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Global Feature Influence
        </div>

        <div class="fs-section-subtitle">
            Mean absolute SHAP values aggregated across the evaluated
            failure-prediction models.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not global_df.empty:
        chart_df = global_df.copy()

        if "mean" in chart_df.columns:
            chart_df = chart_df.sort_values("mean", ascending=True)

            fig = px.bar(
                chart_df,
                x="mean",
                y="feature",
                orientation="h",
                labels={
                    "mean": "Mean |SHAP value|",
                    "feature": "",
                },
                text_auto=".3f",
            )

            fig.update_layout(
                height=max(360, len(chart_df) * 42),
                margin=dict(l=10, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#D9DEE7"),
                xaxis=dict(
                    gridcolor="rgba(255,255,255,0.08)",
                    zeroline=False,
                ),
                yaxis=dict(
                    gridcolor="rgba(0,0,0,0)",
                ),
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        st.caption(
            "Higher mean absolute SHAP values indicate greater average "
            "contribution to model predictions. SHAP magnitude does not "
            "by itself indicate whether a feature increases or decreases risk."
        )

    # ------------------------------------------------------------------
    # Global comparison table
    # ------------------------------------------------------------------

    if not global_df.empty:
        st.markdown(
            """
            <div class="fs-section-title">
                Global SHAP Comparison
            </div>
            """,
            unsafe_allow_html=True,
        )

        display_df = global_df.copy()

        if "mean" in display_df.columns:
            display_df["mean"] = display_df["mean"].round(4)

        if "std" in display_df.columns:
            display_df["std"] = display_df["std"].round(4)

        display_df = display_df.rename(
            columns={
                "feature": "Feature",
                "mean": "Mean |SHAP|",
                "std": "SHAP Std. Dev.",
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

    # ------------------------------------------------------------------
    # Model-specific comparison
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Model-Specific SHAP Evidence
        </div>

        <div class="fs-section-subtitle">
            Comparison of feature influence across Gradient Boosting,
            Random Forest, and XGBoost.
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparison_rows = []

    for model_name, frame in model_frames.items():
        if frame.empty:
            continue

        if "feature" not in frame.columns:
            continue

        value_column = "mean_abs_shap"

        if value_column not in frame.columns:
            continue

        for _, row in frame.iterrows():
            comparison_rows.append(
                {
                    "Model": model_name,
                    "Feature": row["feature"],
                    "Mean |SHAP|": row[value_column],
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)

    if not comparison_df.empty:
        fig = px.bar(
            comparison_df,
            x="Feature",
            y="Mean |SHAP|",
            color="Model",
            barmode="group",
            text_auto=".3f",
        )

        fig.update_layout(
            height=480,
            margin=dict(l=10, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#D9DEE7"),
            xaxis=dict(
                gridcolor="rgba(255,255,255,0.08)",
                tickangle=-25,
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0.08)",
                zeroline=False,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # ------------------------------------------------------------------
    # Individual model inspection
    # ------------------------------------------------------------------

    if available_models:
        selected_model = st.selectbox(
            "Inspect model",
            available_models,
            key="explainability_model",
        )

        selected_df = model_frames[selected_model].copy()

        if not selected_df.empty:
            value_column = "mean_abs_shap"

            if value_column in selected_df.columns:
                selected_df = selected_df.sort_values(
                    value_column,
                    ascending=True,
                )

                fig = px.bar(
                    selected_df,
                    x=value_column,
                    y="feature",
                    orientation="h",
                    labels={
                        value_column: "Mean |SHAP value|",
                        "feature": "",
                    },
                    text_auto=".3f",
                )

                fig.update_layout(
                    height=max(340, len(selected_df) * 42),
                    margin=dict(l=10, r=20, t=20, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#D9DEE7"),
                    xaxis=dict(
                        gridcolor="rgba(255,255,255,0.08)",
                        zeroline=False,
                    ),
                    yaxis=dict(
                        gridcolor="rgba(0,0,0,0)",
                    ),
                    showlegend=False,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    # ------------------------------------------------------------------
    # SHAP visual evidence
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            SHAP Visual Evidence
        </div>

        <div class="fs-section-subtitle">
            Existing SHAP summary and importance artifacts generated by
            the explainability pipeline.
        </div>
        """,
        unsafe_allow_html=True,
    )

    image_pairs = [
        (
            "Gradient Boosting",
            "gradient_boosting_shap_summary.png",
            "gradient_boosting_shap_bar.png",
        ),
        (
            "Random Forest",
            "random_forest_shap_summary.png",
            "random_forest_shap_bar.png",
        ),
        (
            "XGBoost",
            "xgboost_shap_summary.png",
            "xgboost_shap_bar.png",
        ),
    ]

    for model_name, summary_file, bar_file in image_pairs:
        summary_path = _image_path(summary_file)
        bar_path = _image_path(bar_file)

        if summary_path or bar_path:
            st.markdown(
                f"""
                <div style="
                    font-size:16px;
                    font-weight:700;
                    margin:18px 0 10px 0;
                ">
                    {model_name}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                if summary_path:
                    st.image(
                        summary_path,
                        caption=f"{model_name} SHAP Summary",
                        use_container_width=True,
                    )

            with col2:
                if bar_path:
                    st.image(
                        bar_path,
                        caption=f"{model_name} SHAP Importance",
                        use_container_width=True,
                    )

    # ------------------------------------------------------------------
    # Dependence analysis
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Feature Dependence Analysis
        </div>

        <div class="fs-section-subtitle">
            SHAP dependence plots provide feature-level evidence about how
            changes in selected sensor variables relate to model output.
        </div>
        """,
        unsafe_allow_html=True,
    )

    dependence_features = {
        "Gradient Boosting": [
            (
                "Rotational speed",
                "gradient_boosting_dependence_Rotational_speed.png",
            ),
            (
                "Temperature Differential",
                "gradient_boosting_dependence_Temperature_Differential.png",
            ),
            (
                "Tool wear",
                "gradient_boosting_dependence_Tool_wear.png",
            ),
        ],
        "Random Forest": [
            (
                "Mechanical Power",
                "random_forest_dependence_Mechanical_Power.png",
            ),
            (
                "Rotational speed",
                "random_forest_dependence_Rotational_speed.png",
            ),
            (
                "Tool wear",
                "random_forest_dependence_Tool_wear.png",
            ),
        ],
        "XGBoost": [
            (
                "Mechanical Power",
                "xgboost_dependence_Mechanical_Power.png",
            ),
            (
                "Rotational speed",
                "xgboost_dependence_Rotational_speed.png",
            ),
            (
                "Tool wear",
                "xgboost_dependence_Tool_wear.png",
            ),
        ],
    }

    if available_models:
        dependence_model = st.selectbox(
            "Dependence model",
            available_models,
            key="dependence_model",
        )

        available_dependence = [
            (feature, filename)
            for feature, filename in dependence_features.get(
                dependence_model,
                [],
            )
            if (EXPLAINABILITY_DIR / filename).exists()
        ]

        if available_dependence:
            selected_feature = st.selectbox(
                "Feature",
                [feature for feature, _ in available_dependence],
                key="dependence_feature",
            )

            selected_file = dict(available_dependence)[selected_feature]
            selected_path = _image_path(selected_file)

            if selected_path:
                st.image(
                    selected_path,
                    caption=(
                        f"{dependence_model} SHAP dependence: "
                        f"{selected_feature}"
                    ),
                    use_container_width=True,
                )

    # ------------------------------------------------------------------
    # Research interpretation
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Research Interpretation
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fs-card" style="padding:22px 24px; line-height:1.7;">

        <strong>Why explainability is included</strong><br>
        ForgeShield uses explainability to expose the evidence behind
        machine-failure predictions rather than presenting a prediction
        as an unexplained risk signal.

        <br><br>

        <strong>Global evidence</strong><br>
        Global SHAP analysis summarizes which engineered sensor features
        contribute most strongly to model predictions across the evaluated
        samples.

        <br><br>

        <strong>Model-specific evidence</strong><br>
        Comparing SHAP importance across different tree-based models shows
        whether important predictive features remain consistent across
        model architectures.

        <br><br>

        <strong>Feature-level evidence</strong><br>
        Dependence plots provide additional evidence about the relationship
        between individual sensor variables and their contribution to the
        predicted failure output.

        <br><br>

        <strong>Safety interpretation</strong><br>
        SHAP values explain model behavior. They should not be interpreted
        as causal proof that changing a particular sensor variable will
        itself cause or prevent equipment failure.

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # Methodology
    # ------------------------------------------------------------------

    st.markdown(
        """
        <div class="fs-section-title">
            Explainability Methodology
        </div>
        """,
        unsafe_allow_html=True,
    )

    methodology_cols = st.columns(3)

    with methodology_cols[0]:
        st.markdown(
            """
            <div class="fs-card" style="padding:18px;">
                <div style="font-weight:700; margin-bottom:8px;">
                    SHAP
                </div>
                <div style="color:#AAB2BF; font-size:13px; line-height:1.6;">
                    Shapley-value-based attribution is used to quantify
                    feature contributions to individual model predictions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with methodology_cols[1]:
        st.markdown(
            """
            <div class="fs-card" style="padding:18px;">
                <div style="font-weight:700; margin-bottom:8px;">
                    Global Analysis
                </div>
                <div style="color:#AAB2BF; font-size:13px; line-height:1.6;">
                    Mean absolute SHAP values summarize feature influence
                    across the evaluated prediction samples.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with methodology_cols[2]:
        st.markdown(
            """
            <div class="fs-card" style="padding:18px;">
                <div style="font-weight:700; margin-bottom:8px;">
                    Local Evidence
                </div>
                <div style="color:#AAB2BF; font-size:13px; line-height:1.6;">
                    Model-specific SHAP artifacts provide additional
                    evidence for interpreting individual feature effects.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="
            margin-top:24px;
            padding-top:18px;
            border-top:1px solid rgba(255,255,255,0.08);
            color:#69717F;
            font-size:12px;
            display:flex;
            justify-content:space-between;
        ">
            <span>FORGESHIELD · EXPLAINABILITY</span>
            <span>Research Proof of Concept · SHAP Evidence</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
