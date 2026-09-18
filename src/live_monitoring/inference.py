from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models" / "supervised"
ANOMALY_DIR = PROJECT_ROOT / "models" / "anomaly"
AI4I_DIR = PROJECT_ROOT / "data" / "processed" / "ai4i"


class ForgeShieldLiveInference:
    """
    Live inference layer for the ForgeShield research POC.

    Raw machine telemetry is transformed using the exact
    preprocessing artifacts generated during model training,
    then passed through the existing Gradient Boosting and
    Isolation Forest models.
    """

    def __init__(self):

        # =========================================================
        # Load trained models
        # =========================================================

        supervised_path = (
            MODEL_DIR / "gradient_boosting.joblib"
        )

        anomaly_path = (
            ANOMALY_DIR / "isolation_forest.joblib"
        )

        if not supervised_path.exists():
            raise FileNotFoundError(
                f"Gradient Boosting model not found: "
                f"{supervised_path}"
            )

        if not anomaly_path.exists():
            raise FileNotFoundError(
                f"Isolation Forest model not found: "
                f"{anomaly_path}"
            )

        self.supervised_model = joblib.load(
            supervised_path
        )

        self.anomaly_model = joblib.load(
            anomaly_path
        )

        # =========================================================
        # Load EXACT preprocessing artifacts
        # =========================================================

        scaler_path = AI4I_DIR / "scaler.joblib"
        encoder_path = AI4I_DIR / "encoder.joblib"
        metadata_path = (
            AI4I_DIR / "preprocessing_metadata.joblib"
        )

        for path in [
            scaler_path,
            encoder_path,
            metadata_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Required preprocessing artifact not found: "
                    f"{path}"
                )

        self.scaler = joblib.load(
            scaler_path
        )

        self.encoder = joblib.load(
            encoder_path
        )

        self.preprocessing_metadata = joblib.load(
            metadata_path
        )

        # =========================================================
        # Exact feature schema
        # =========================================================

        self.numerical_features = (
            self.preprocessing_metadata[
                "numerical_features"
            ]
        )

        self.categorical_features = (
            self.preprocessing_metadata[
                "categorical_features"
            ]
        )

        self.feature_columns = (
            self.preprocessing_metadata[
                "feature_columns"
            ]
        )

        self.clipping_bounds = (
            self.preprocessing_metadata[
                "clipping_bounds"
            ]
        )

        # =========================================================
        # Verify model feature schema
        # =========================================================

        self.model_feature_names = list(
            getattr(
                self.supervised_model,
                "feature_names_in_",
                self.feature_columns,
            )
        )

        if self.model_feature_names != self.feature_columns:
            raise ValueError(
                "Model feature schema does not match "
                "preprocessing feature schema.\n"
                f"Model: {self.model_feature_names}\n"
                f"Preprocessing: {self.feature_columns}"
            )

    # =============================================================
    # Feature engineering
    # =============================================================

    @staticmethod
    def _engineer_features(
        telemetry: pd.DataFrame,
    ) -> pd.DataFrame:

        data = telemetry.copy()

        data["Temperature Differential"] = (
            data["Process temperature"]
            - data["Air temperature"]
        )

        data["Mechanical Power"] = (
            data["Torque"]
            * data["Rotational speed"]
            * (2 * np.pi / 60)
        )

        return data

    # =============================================================
    # Apply training-time IQR clipping
    # =============================================================

    def _clip_outliers(
        self,
        telemetry: pd.DataFrame,
    ) -> pd.DataFrame:

        data = telemetry.copy()

        for feature, bounds in (
            self.clipping_bounds.items()
        ):

            if feature not in data.columns:
                continue

            lower, upper = bounds

            data[feature] = data[
                feature
            ].clip(
                lower,
                upper,
            )

        return data

    # =============================================================
    # Prepare exact model matrix
    # =============================================================

    def _prepare_features(
        self,
        telemetry: pd.DataFrame,
    ) -> pd.DataFrame:

        data = self._engineer_features(
            telemetry
        )

        data = self._clip_outliers(
            data
        )

        # ---------------------------------------------------------
        # Standardize numerical features using the EXACT scaler
        # fitted during training.
        # ---------------------------------------------------------

        numerical_values = (
            self.scaler.transform(
                data[self.numerical_features]
            )
        )

        numerical_df = pd.DataFrame(
            numerical_values,
            columns=self.numerical_features,
            index=data.index,
        )

        # ---------------------------------------------------------
        # One-hot encode Type using the EXACT encoder
        # fitted during training.
        # ---------------------------------------------------------

        categorical_values = (
            self.encoder.transform(
                data[self.categorical_features]
            )
        )

        categorical_columns = (
            self.encoder
            .get_feature_names_out(
                self.categorical_features
            )
            .tolist()
        )

        categorical_df = pd.DataFrame(
            categorical_values,
            columns=categorical_columns,
            index=data.index,
        )

        # ---------------------------------------------------------
        # Combine exactly as the training pipeline did.
        # ---------------------------------------------------------

        X = pd.concat(
            [
                numerical_df,
                categorical_df,
            ],
            axis=1,
        )

        X = X[self.feature_columns]

        return X.astype(float)

    # =============================================================
    # Isolation Forest anomaly score
    # =============================================================

    def _calculate_anomaly_score(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:

        decision_scores = (
            self.anomaly_model
            .decision_function(X)
        )

        # Isolation Forest:
        # lower decision score = more anomalous.
        #
        # Invert it so:
        # higher anomaly score = greater risk.
        anomaly_signal = -decision_scores

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # For a live fleet, scores should be comparable across
        # machines. Therefore use a reference distribution from
        # the training data instead of min-max normalizing the
        # current six machines against each other.
        # ---------------------------------------------------------

        training_path = (
            AI4I_DIR / "X_train.csv"
        )

        X_train = pd.read_csv(
            training_path
        )

        training_decision_scores = (
            self.anomaly_model
            .decision_function(
                X_train[
                    self.model_feature_names
                ]
            )
        )

        training_anomaly_signal = (
            -training_decision_scores
        )

        # Robust reference range based on the training
        # distribution.
        lower = np.percentile(
            training_anomaly_signal,
            5,
        )

        upper = np.percentile(
            training_anomaly_signal,
            95,
        )

        if upper <= lower:
            return np.zeros_like(
                anomaly_signal,
                dtype=float,
            )

        anomaly_score = (
            anomaly_signal - lower
        ) / (
            upper - lower
        )

        return np.clip(
            anomaly_score,
            0.0,
            1.0,
        )

    # =============================================================
    # Risk band
    # =============================================================

    @staticmethod
    def _risk_band(
        score: float,
    ) -> str:

        if score < 0.25:
            return "Low"

        if score < 0.50:
            return "Medium"

        if score < 0.75:
            return "High"

        return "Critical"

    # =============================================================
    # Main prediction method
    # =============================================================

    def predict(
        self,
        telemetry: pd.DataFrame,
    ) -> pd.DataFrame:

        if telemetry.empty:
            return telemetry.copy()

        # ---------------------------------------------------------
        # Transform raw telemetry into model features
        # ---------------------------------------------------------

        X = self._prepare_features(
            telemetry
        )

        # ---------------------------------------------------------
        # Supervised failure probability
        # ---------------------------------------------------------

        failure_probability = (
            self.supervised_model
            .predict_proba(X)[:, 1]
        )

        # ---------------------------------------------------------
        # Unsupervised anomaly score
        # ---------------------------------------------------------

        anomaly_score = (
            self._calculate_anomaly_score(X)
        )

        # ---------------------------------------------------------
        # Unified ForgeShield risk score
        #
        # 60% supervised failure probability
        # 40% unsupervised anomaly score
        # ---------------------------------------------------------

        unified_risk = (
            0.60 * failure_probability
            + 0.40 * anomaly_score
        )

        # ---------------------------------------------------------
        # Preserve original telemetry
        # ---------------------------------------------------------

        result = telemetry.copy()

        result["failure_probability"] = (
            failure_probability
        )

        result["anomaly_score"] = (
            anomaly_score
        )

        result["unified_risk_score"] = (
            unified_risk
        )

        result["risk_band"] = [
            self._risk_band(score)
            for score in unified_risk
        ]

        return result


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("FORGESHIELD | LIVE INFERENCE TEST")
    print("=" * 70)

    # -------------------------------------------------------------
    # Import simulator
    # -------------------------------------------------------------

    from simulator import (
        create_machine_fleet,
        generate_fleet_snapshot,
    )

    # -------------------------------------------------------------
    # Create simulated machine fleet
    # -------------------------------------------------------------

    fleet = create_machine_fleet(
        machine_count=6,
        seed=42,
    )

    telemetry = generate_fleet_snapshot(
        fleet
    )

    print("\n=== RAW LIVE TELEMETRY ===")

    print(
        telemetry[
            [
                "machine_id",
                "Type",
                "Air temperature",
                "Process temperature",
                "Rotational speed",
                "Torque",
                "Tool wear",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------
    # Run ForgeShield inference
    # -------------------------------------------------------------

    inference = ForgeShieldLiveInference()

    results = inference.predict(
        telemetry
    )

    print(
        "\n=== FORGESHIELD LIVE INFERENCE ==="
    )

    print(
        results[
            [
                "machine_id",
                "failure_probability",
                "anomaly_score",
                "unified_risk_score",
                "risk_band",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("LIVE INFERENCE TEST COMPLETE")
    print("=" * 70)
