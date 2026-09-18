from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ai4i"
    / "ai4i2020.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ai4i"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


NUMERICAL_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Temperature Differential",
    "Mechanical Power",
]

CATEGORICAL_FEATURES = [
    "Type",
]

TARGET = "Machine failure"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

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


def clip_outliers(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train = train_df.copy()
    test = test_df.copy()

    for column in columns:
        q1 = train[column].quantile(0.25)
        q3 = train[column].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        train[column] = train[column].clip(lower, upper)
        test[column] = test[column].clip(lower, upper)

    return train, test


def run_preprocessing() -> None:

    print("\n" + "=" * 60)
    print("FORGESHIELD | AI4I PREPROCESSING")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print(f"\nOriginal dataset shape: {df.shape}")

    # ---------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------

    df = engineer_features(df)

    # ---------------------------------------------------------
    # Select modeling variables
    # ---------------------------------------------------------

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    print(f"Feature matrix shape before encoding: {X.shape}")
    print(f"Target shape: {y.shape}")

    # ---------------------------------------------------------
    # Train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # ---------------------------------------------------------
    # Outlier clipping
    # Learn thresholds ONLY from training data
    # ---------------------------------------------------------

    X_train, X_test = clip_outliers(
        X_train,
        X_test,
        NUMERICAL_FEATURES,
    )

    # ---------------------------------------------------------
    # Standardize numerical features
    # Fit scaler ONLY on training data
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_train_numeric = scaler.fit_transform(
        X_train[NUMERICAL_FEATURES]
    )

    X_test_numeric = scaler.transform(
        X_test[NUMERICAL_FEATURES]
    )

    # ---------------------------------------------------------
    # One-hot encode Type
    # Fit encoder ONLY on training data
    # ---------------------------------------------------------

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )

    X_train_type = encoder.fit_transform(
        X_train[CATEGORICAL_FEATURES]
    )

    X_test_type = encoder.transform(
        X_test[CATEGORICAL_FEATURES]
    )

    type_feature_names = encoder.get_feature_names_out(
        CATEGORICAL_FEATURES
    ).tolist()

    # ---------------------------------------------------------
    # Combine numerical + categorical features
    # ---------------------------------------------------------

    FEATURES = NUMERICAL_FEATURES + type_feature_names

    X_train_final = np.hstack(
        [X_train_numeric, X_train_type]
    )

    X_test_final = np.hstack(
        [X_test_numeric, X_test_type]
    )

    X_train_final = pd.DataFrame(
        X_train_final,
        columns=FEATURES,
        index=X_train.index,
    )

    X_test_final = pd.DataFrame(
        X_test_final,
        columns=FEATURES,
        index=X_test.index,
    )

    print(f"\nFinal feature matrix shape: {X_train_final.shape}")
    print("\nFinal features:")

    for feature in FEATURES:
        print(f"  ✓ {feature}")

     #-------------------------------------------------------
    # Preserve original test-set metadata
    # ---------------------------------------------------------

    test_metadata_columns = [
        column
        for column in [
            "UID",
            "Type",
            "Air temperature",
            "Process temperature",
            "Rotational speed",
            "Torque",
            "Tool wear",
            "Machine failure",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF",
        ]
        if column in df.columns
    ]

    test_metadata = df.loc[
        X_test.index,
        test_metadata_columns,
    ].copy()

    test_metadata.to_csv(
        OUTPUT_DIR / "test_metadata.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Save datasets
    # ---------------------------------------------------------

    X_train_final.to_csv(
        OUTPUT_DIR / "X_train.csv",
        index=False,
    )

    X_test_final.to_csv(
        OUTPUT_DIR / "X_test.csv",
        index=False,
    )

    y_train.to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False,
    )

    y_test.to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Save feature metadata
    # ---------------------------------------------------------

    feature_types = (
        ["original"] * 5
        + ["engineered"] * 2
        + ["categorical_encoded"] * len(type_feature_names)
    )

    pd.DataFrame(
        {
            "feature": FEATURES,
            "type": feature_types,
        }
    ).to_csv(
        OUTPUT_DIR / "feature_metadata.csv",
        index=False,
    )

    print("\nProcessed files created:")

    for file in sorted(OUTPUT_DIR.iterdir()):
        print(f"  ✓ {file.name}")

    print("\nTraining failure rate:")
    print(f"{y_train.mean() * 100:.2f}%")

    print("Testing failure rate:")
    print(f"{y_test.mean() * 100:.2f}%")

    print("\n" + "=" * 60)
    print("AI4I PREPROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()