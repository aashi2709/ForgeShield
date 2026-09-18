from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cmapss"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "cmapss"

TRAIN_PATH = RAW_DIR / "train_FD001.txt"
TEST_PATH = RAW_DIR / "test_FD001.txt"
RUL_PATH = RAW_DIR / "RUL_FD001.txt"


COLUMN_NAMES = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


WINDOW_SIZE = 30
RUL_CAP = 125


def load_cmapss_file(path: Path) -> pd.DataFrame:

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )


def calculate_training_rul(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    max_cycles = (
        df.groupby("unit_id")["cycle"]
        .transform("max")
    )

    df["RUL"] = np.minimum(
        max_cycles - df["cycle"],
        RUL_CAP,
    )

    return df


def calculate_test_rul(
    df: pd.DataFrame,
    rul_values: pd.Series,
) -> pd.DataFrame:

    df = df.copy()

    max_cycles = (
        df.groupby("unit_id")["cycle"]
        .max()
        .sort_index()
    )

    rul_values = rul_values.reset_index(
        drop=True
    )

    if len(max_cycles) != len(rul_values):

        raise ValueError(
            "Number of test engines does not match "
            "number of RUL labels."
        )

    final_rul = pd.Series(
        rul_values.values,
        index=max_cycles.index,
    )

    df["RUL"] = np.minimum(
        df["unit_id"].map(max_cycles)
        + df["unit_id"].map(final_rul)
        - df["cycle"],
        RUL_CAP,
    )

    return df


def create_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    window_size: int,
):

    X = []
    y = []
    units = []

    for unit_id, group in df.groupby("unit_id"):

        group = group.sort_values("cycle")

        features = group[
            feature_columns
        ].to_numpy(
            dtype=np.float32
        )

        targets = group[
            "RUL"
        ].to_numpy(
            dtype=np.float32
        )

        if len(group) < window_size:
            continue

        for end in range(
            window_size,
            len(group) + 1,
        ):

            start = end - window_size

            X.append(
                features[start:end]
            )

            y.append(
                targets[end - 1]
            )

            units.append(
                unit_id
            )

    return (
        np.asarray(
            X,
            dtype=np.float32,
        ),
        np.asarray(
            y,
            dtype=np.float32,
        ),
        np.asarray(units),
    )


def main():

    print("\n" + "=" * 70)
    print(
        "FORGESHIELD | C-MAPSS PREPROCESSING"
    )
    print("=" * 70)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("\nLoading datasets...")

    train = load_cmapss_file(
        TRAIN_PATH
    )

    test = load_cmapss_file(
        TEST_PATH
    )

    rul = pd.read_csv(
        RUL_PATH,
        header=None,
        names=["RUL"],
    )["RUL"]

    print(
        f"  Train rows: {len(train)}"
    )

    print(
        f"  Test rows : {len(test)}"
    )

    print(
        f"  RUL labels: {len(rul)}"
    )

    # ---------------------------------------------------------
    # Calculate capped RUL
    # ---------------------------------------------------------

    print(
        f"\nCalculating RUL targets "
        f"(cap = {RUL_CAP} cycles)..."
    )

    train = calculate_training_rul(
        train
    )

    test = calculate_test_rul(
        test,
        rul,
    )

    print(
        f"  Training RUL range: "
        f"{train['RUL'].min()} - "
        f"{train['RUL'].max()}"
    )

    print(
        f"  Testing RUL range: "
        f"{test['RUL'].min()} - "
        f"{test['RUL'].max()}"
    )

    print(
        f"  Training values at cap: "
        f"{(train['RUL'] == RUL_CAP).sum()}"
    )

    print(
        f"  Testing values at cap: "
        f"{(test['RUL'] == RUL_CAP).sum()}"
    )

    # ---------------------------------------------------------
    # Feature columns
    # ---------------------------------------------------------

    feature_columns = [
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
        *[
            f"sensor_{i}"
            for i in range(1, 22)
        ],
    ]

    # ---------------------------------------------------------
    # Scale using TRAINING DATA ONLY
    # ---------------------------------------------------------

    print(
        "\nScaling features..."
    )

    scaler = StandardScaler()

    train[feature_columns] = (
        train[feature_columns]
        .astype(np.float64)
    )

    test[feature_columns] = (
        test[feature_columns]
        .astype(np.float64)
    )

    train_features = (
        train[feature_columns]
        .to_numpy(dtype=np.float32)
    )

    test_features = (
        test[feature_columns]
        .to_numpy(dtype=np.float32)
    )

    scaler.fit(
        train_features
    )

    train.loc[:, feature_columns] = (
        scaler.transform(
            train_features
        )
    )

    test.loc[:, feature_columns] = (
        scaler.transform(
            test_features
        )
    )

    # ---------------------------------------------------------
    # Create sliding windows
    # ---------------------------------------------------------

    print(
        f"\nCreating "
        f"{WINDOW_SIZE}-cycle sliding windows..."
    )

    X_train, y_train, train_units = (
        create_windows(
            train,
            feature_columns,
            WINDOW_SIZE,
        )
    )

    X_test, y_test, test_units = (
        create_windows(
            test,
            feature_columns,
            WINDOW_SIZE,
        )
    )

    print(
        "\nWINDOWED DATA"
    )

    print(
        f"  X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"  y_train shape: "
        f"{y_train.shape}"
    )

    print(
        f"  X_test shape : "
        f"{X_test.shape}"
    )

    print(
        f"  y_test shape : "
        f"{y_test.shape}"
    )

    # ---------------------------------------------------------
    # Save arrays
    # ---------------------------------------------------------

    np.save(
        PROCESSED_DIR / "X_train.npy",
        X_train,
    )

    np.save(
        PROCESSED_DIR / "y_train.npy",
        y_train,
    )

    np.save(
        PROCESSED_DIR / "train_units.npy",
        train_units,
    )

    np.save(
        PROCESSED_DIR / "X_test.npy",
        X_test,
    )

    np.save(
        PROCESSED_DIR / "y_test.npy",
        y_test,
    )

    np.save(
        PROCESSED_DIR / "test_units.npy",
        test_units,
    )

    # ---------------------------------------------------------
    # Save scaler
    # ---------------------------------------------------------

    joblib.dump(
        scaler,
        PROCESSED_DIR / "scaler.joblib",
    )

    # ---------------------------------------------------------
    # Save feature metadata
    # ---------------------------------------------------------

    metadata = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_type": [
                (
                    "operational_setting"
                    if feature.startswith(
                        "op_setting"
                    )
                    else "sensor"
                )
                for feature in feature_columns
            ],
        }
    )

    metadata.to_csv(
        PROCESSED_DIR
        / "feature_metadata.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Save preprocessing metadata
    # ---------------------------------------------------------

    preprocessing_metadata = pd.DataFrame(
        {
            "parameter": [
                "dataset",
                "window_size",
                "rul_cap",
                "num_features",
                "train_engines",
                "test_engines",
            ],
            "value": [
                "C-MAPSS FD001",
                WINDOW_SIZE,
                RUL_CAP,
                len(feature_columns),
                train["unit_id"].nunique(),
                test["unit_id"].nunique(),
            ],
        }
    )

    preprocessing_metadata.to_csv(
        PROCESSED_DIR
        / "preprocessing_metadata.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\nSaved files:")

    for path in sorted(
        PROCESSED_DIR.iterdir()
    ):

        print(
            f"  ✓ {path.name}"
        )

    print("\n" + "=" * 70)
    print(
        "C-MAPSS PREPROCESSING COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()