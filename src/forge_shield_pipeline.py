from pathlib import Path
import json

import numpy as np
import pandas as pd

from risk_scoring import (
    load_ai4i_data,
    load_supervised_model,
    load_isolation_forest,
    calculate_anomaly_score,
    calculate_risk_scores,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


TEST_METADATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ai4i"
    / "test_metadata.csv"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "integration"
)


class ForgeShieldPipeline:

    def __init__(self):

        print("=" * 70)
        print("FORGESHIELD INTEGRATED INTELLIGENCE PIPELINE")
        print("=" * 70)

        # -----------------------------------------------------
        # Load processed AI4I data
        # -----------------------------------------------------

        print("\nLoading AI4I data...")

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = load_ai4i_data()

        print(
            f"Training data: {self.X_train.shape}"
        )

        print(
            f"Test data: {self.X_test.shape}"
        )

        # -----------------------------------------------------
        # Load trained models
        # -----------------------------------------------------

        print("\nLoading trained models...")

        self.supervised_model = (
            load_supervised_model()
        )

        self.anomaly_model = (
            load_isolation_forest()
        )

    # =========================================================
    # MODEL INPUT
    # =========================================================

    @staticmethod
    def prepare_model_features(
        X,
        model,
    ):

        """
        Prepare the processed feature matrix for a trained
        scikit-learn model.

        The preprocessing stage already produces the
        model-ready numerical matrix, so no additional
        scaling or encoding is performed here.
        """

        X_model = X.copy()

        removable_columns = [
            "UID",
            "UDI",
            "Machine failure",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF",
        ]

        X_model = X_model.drop(
            columns=[
                column
                for column in removable_columns
                if column in X_model.columns
            ],
            errors="ignore",
        )

        # Align columns to the features expected by the
        # trained estimator when feature names are available.
        if hasattr(
            model,
            "feature_names_in_",
        ):

            expected_features = list(
                model.feature_names_in_
            )

            missing = [
                feature
                for feature in expected_features
                if feature not in X_model.columns
            ]

            if missing:

                raise ValueError(
                    "The model expects features that are "
                    f"missing from X_test: {missing}"
                )

            X_model = X_model[
                expected_features
            ]

        return X_model

    # =========================================================
    # SUPERVISED FAILURE PREDICTION
    # =========================================================

    def predict_failure_probability(self):

        """
        Generate failure probabilities for every AI4I
        test observation using the trained Gradient Boosting
        classifier.
        """

        X_model = (
            self.prepare_model_features(
                self.X_test,
                self.supervised_model,
            )
        )

        probabilities = (
            self.supervised_model
            .predict_proba(X_model)[:, 1]
        )

        return probabilities

    # =========================================================
    # ANOMALY DETECTION
    # =========================================================

    def calculate_anomaly_scores(self):

        """
        Generate continuous Isolation Forest anomaly
        scores for the complete AI4I test set.

        The complete test set must be supplied because the
        existing ForgeShield implementation min-max normalizes
        the anomaly scores.
        """

        X_model = (
            self.prepare_model_features(
                self.X_test,
                self.anomaly_model,
            )
        )

        anomaly_scores = (
            calculate_anomaly_score(
                self.anomaly_model,
                X_model,
            )
        )

        return anomaly_scores

    # =========================================================
    # UNIFIED RISK
    # =========================================================

    def build_risk_table(self):

        """
        Fuse supervised failure probability and anomaly
        score using the existing ForgeShield risk-scoring
        implementation.
        """

        print(
            "\nGenerating supervised failure probabilities..."
        )

        failure_probability = (
            self.predict_failure_probability()
        )

        print(
            "Generating anomaly scores..."
        )

        anomaly_score = (
            self.calculate_anomaly_scores()
        )

        print(
            "Calculating unified risk scores..."
        )

        unified_risk, risk_bands = (
            calculate_risk_scores(
                failure_probability,
                anomaly_score,
            )
        )

        risk_df = self.X_test.copy()

        risk_df[
            "failure_probability"
        ] = failure_probability

        risk_df[
            "anomaly_score"
        ] = anomaly_score

        risk_df[
            "unified_risk_score"
        ] = unified_risk

        risk_df[
            "risk_band"
        ] = risk_bands

        risk_df[
            "actual_failure"
        ] = self.y_test.values

        return risk_df

    # =========================================================
    # RISK BAND
    # =========================================================

    @staticmethod
    def calculate_risk_band(
        risk_score,
    ):

        if risk_score < 0.25:

            return "Low"

        elif risk_score < 0.50:

            return "Medium"

        elif risk_score < 0.75:

            return "High"

        return "Critical"

    # =========================================================
    # TEST METADATA
    # =========================================================

    def load_test_metadata(self):

        """
        Load the original AI4I metadata corresponding to
        the processed test observations.
        """

        if not TEST_METADATA.exists():

            raise FileNotFoundError(
                "Test metadata file not found:\n"
                f"{TEST_METADATA}"
            )

        metadata = pd.read_csv(
            TEST_METADATA
        )

        if len(metadata) != len(self.X_test):

            raise ValueError(
                "Test metadata length does not match X_test.\n"
                f"Metadata rows: {len(metadata)}\n"
                f"X_test rows: {len(self.X_test)}"
            )

        return metadata

    # =========================================================
    # SELECT DEMO EVENT
    # =========================================================

    @staticmethod
    def select_demo_event(
        risk_df,
    ):

        """
        Select the highest-risk test observation for the
        integration demonstration.

        This does not modify model training or evaluation.
        """

        index = (
            risk_df[
                "unified_risk_score"
            ]
            .idxmax()
        )

        row = (
            risk_df.loc[index]
        )

        return index, row

    # =========================================================
    # EVENT TYPE
    # =========================================================

    @staticmethod
    def determine_event_type(
        metadata_row,
    ):

        """
        Map AI4I failure-mode indicators to an interpretable
        event type.

        The order follows the failure-mode columns in the
        AI4I dataset.
        """

        if metadata_row is None:

            return "risk_detected"

        mapping = [
            (
                "TWF",
                "tool_wear_failure",
            ),
            (
                "HDF",
                "heat_dissipation_failure",
            ),
            (
                "PWF",
                "power_failure",
            ),
            (
                "OSF",
                "overstrain_failure",
            ),
            (
                "RNF",
                "random_failure",
            ),
        ]

        for column, event_type in mapping:

            if column not in metadata_row.index:

                continue

            try:

                if int(
                    metadata_row[column]
                ) == 1:

                    return event_type

            except (
                TypeError,
                ValueError,
            ):

                continue

        return "risk_detected"

    # =========================================================
    # STRUCTURED EVENT
    # =========================================================

    def build_event(
        self,
        risk_row,
        metadata_row,
    ):

        """
        Convert the unified risk result and original AI4I
        metadata into the structured event passed to Part 02.
        """

        risk_score = float(
            risk_row[
                "unified_risk_score"
            ]
        )

        # -----------------------------------------------------
        # Machine identifier
        # -----------------------------------------------------

        machine_id = "UNKNOWN"

        if (
            metadata_row is not None
            and "UID" in metadata_row.index
        ):

            machine_id = str(
                metadata_row["UID"]
            )

        # -----------------------------------------------------
        # Event type
        # -----------------------------------------------------

        event_type = (
            self.determine_event_type(
                metadata_row
            )
        )

        # -----------------------------------------------------
        # Core risk event
        # -----------------------------------------------------

        event = {

            "machine_id":
                machine_id,

            "event_type":
                event_type,

            "risk_score":
                round(
                    risk_score,
                    6,
                ),

            "risk_band":
                self.calculate_risk_band(
                    risk_score
                ),

            "failure_probability":
                round(
                    float(
                        risk_row[
                            "failure_probability"
                        ]
                    ),
                    6,
                ),

            "anomaly_score":
                round(
                    float(
                        risk_row[
                            "anomaly_score"
                        ]
                    ),
                    6,
                ),
        }

        # =====================================================
        # ORIGINAL SENSOR VALUES
        # =====================================================

        if metadata_row is not None:

            sensor_mapping = {

                "Air temperature":
                    "air_temperature",

                "Process temperature":
                    "process_temperature",

                "Rotational speed":
                    "rotational_speed",

                "Torque":
                    "torque",

                "Tool wear":
                    "tool_wear",
            }

            for source, target in (
                sensor_mapping.items()
            ):

                if source not in metadata_row.index:

                    continue

                value = (
                    metadata_row[source]
                )

                if pd.notna(value):

                    event[target] = float(
                        value
                    )

            # -------------------------------------------------
            # Engineered temperature differential
            # -------------------------------------------------

            if (
                "Air temperature"
                in metadata_row.index
                and
                "Process temperature"
                in metadata_row.index
            ):

                temperature_differential = (
                    float(
                        metadata_row[
                            "Process temperature"
                        ]
                    )
                    -
                    float(
                        metadata_row[
                            "Air temperature"
                        ]
                    )
                )

                event[
                    "temperature_differential"
                ] = round(
                    temperature_differential,
                    6,
                )

            # -------------------------------------------------
            # Engineered mechanical power
            # -------------------------------------------------

            if (
                "Torque"
                in metadata_row.index
                and
                "Rotational speed"
                in metadata_row.index
            ):

                mechanical_power = (
                    float(
                        metadata_row[
                            "Torque"
                        ]
                    )
                    *
                    float(
                        metadata_row[
                            "Rotational speed"
                        ]
                    )
                    *
                    (
                        2
                        * np.pi
                        / 60
                    )
                )

                event[
                    "mechanical_power"
                ] = round(
                    mechanical_power,
                    6,
                )

            # -------------------------------------------------
            # Original product type
            # -------------------------------------------------

            if (
                "Type"
                in metadata_row.index
            ):

                event[
                    "product_type"
                ] = str(
                    metadata_row[
                        "Type"
                    ]
                )

            # -------------------------------------------------
            # Original failure-mode indicators
            # -------------------------------------------------

            failure_modes = [
                "TWF",
                "HDF",
                "PWF",
                "OSF",
                "RNF",
            ]

            for mode in failure_modes:

                if mode in metadata_row.index:

                    try:

                        event[
                            mode
                        ] = int(
                            metadata_row[
                                mode
                            ]
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        pass

            # -------------------------------------------------
            # Actual failure label
            # -------------------------------------------------

            if (
                "Machine failure"
                in metadata_row.index
            ):

                try:

                    event[
                        "actual_failure"
                    ] = int(
                        metadata_row[
                            "Machine failure"
                        ]
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    pass

        return event

    # =========================================================
    # RUN
    # =========================================================

    def run(self):

        # =====================================================
        # STEP 1
        # =====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "STEP 1 | MODEL + ANOMALY FUSION"
        )

        print(
            "=" * 70
        )

        risk_df = (
            self.build_risk_table()
        )

        # -----------------------------------------------------
        # Risk distribution
        # -----------------------------------------------------

        print(
            "\nRisk distribution:"
        )

        print(
            risk_df[
                "risk_band"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

        # -----------------------------------------------------
        # Top five
        # -----------------------------------------------------

        print(
            "\nTop 5 risk observations:"
        )

        top5 = (
            risk_df
            .sort_values(
                "unified_risk_score",
                ascending=False,
            )
            .head(5)
        )

        columns_to_show = [
            column
            for column in [
                "failure_probability",
                "anomaly_score",
                "unified_risk_score",
                "risk_band",
                "actual_failure",
            ]
            if column in top5.columns
        ]

        print(
            top5[
                columns_to_show
            ].to_string(
                index=False
            )
        )

        # =====================================================
        # STEP 2
        # =====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "STEP 2 | STRUCTURED EVENT CREATION"
        )

        print(
            "=" * 70
        )

        # -----------------------------------------------------
        # Highest-risk observation
        # -----------------------------------------------------

        index, risk_row = (
            self.select_demo_event(
                risk_df
            )
        )

        print(
            f"\nSelected test observation index: {index}"
        )

        # -----------------------------------------------------
        # Load exact corresponding metadata
        # -----------------------------------------------------

        test_metadata = (
            self.load_test_metadata()
        )

        metadata_row = (
            test_metadata.iloc[index]
        )

        # -----------------------------------------------------
        # Build structured event
        # -----------------------------------------------------

        event = (
            self.build_event(
                risk_row=risk_row,
                metadata_row=metadata_row,
            )
        )

        print(
            "\nStructured ForgeShield event:"
        )

        print(
            json.dumps(
                event,
                indent=4,
            )
        )

        # =====================================================
        # STEP 3
        # =====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "STEP 3 | SAVE INTEGRATION ARTIFACTS"
        )

        print(
            "=" * 70
        )

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Structured event
        # -----------------------------------------------------

        event_path = (
            OUTPUT_DIR
            / "demo_event.json"
        )

        event_path.write_text(
            json.dumps(
                event,
                indent=4,
            ),
            encoding="utf-8",
        )

        # -----------------------------------------------------
        # Complete risk table
        # -----------------------------------------------------

        risk_output_path = (
            OUTPUT_DIR
            / "integrated_risk_table.csv"
        )

        risk_df.to_csv(
            risk_output_path,
            index=False,
        )

        # =====================================================
        # COMPLETE
        # =====================================================

        print(
            "\n" + "=" * 70
        )

        print(
            "INTEGRATION COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            "\nSaved event:"
        )

        print(
            event_path
        )

        print(
            "\nSaved risk table:"
        )

        print(
            risk_output_path
        )

        print(
            "\nPipeline:"
        )

        print(
            "AI4I → Gradient Boosting → "
            "Isolation Forest → Unified Risk → "
            "Structured Event → Part 02"
        )

        return event


def main():

    pipeline = (
        ForgeShieldPipeline()
    )

    pipeline.run()


if __name__ == "__main__":

    main()