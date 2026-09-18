from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cmapss"
)

TRAIN_PATH = DATA_DIR / "train_FD001.txt"
TEST_PATH = DATA_DIR / "test_FD001.txt"
RUL_PATH = DATA_DIR / "RUL_FD001.txt"


COLUMN_NAMES = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


def load_cmapss_file(path: Path) -> pd.DataFrame:

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )


def run_validation():

    print("\n" + "=" * 70)
    print("FORGESHIELD | C-MAPSS FD001 VALIDATION")
    print("=" * 70)

    train = load_cmapss_file(TRAIN_PATH)
    test = load_cmapss_file(TEST_PATH)

    rul = pd.read_csv(
        RUL_PATH,
        header=None,
        names=["RUL"],
    )

    # ---------------------------------------------------------
    # Basic shapes
    # ---------------------------------------------------------

    print("\nDATASET SHAPES")

    print(f"  Training data : {train.shape}")
    print(f"  Testing data  : {test.shape}")
    print(f"  RUL data      : {rul.shape}")

    # ---------------------------------------------------------
    # Expected columns
    # ---------------------------------------------------------

    print("\nCOLUMN COUNT")

    print(f"  Total columns : {len(COLUMN_NAMES)}")
    print("  Expected      : 26")

    # ---------------------------------------------------------
    # Unique engines
    # ---------------------------------------------------------

    print("\nUNIQUE ENGINES")

    print(
        f"  Training engines : "
        f"{train['unit_id'].nunique()}"
    )

    print(
        f"  Testing engines  : "
        f"{test['unit_id'].nunique()}"
    )

    print(
        f"  RUL values       : "
        f"{len(rul)}"
    )

    # ---------------------------------------------------------
    # Cycle information
    # ---------------------------------------------------------

    train_cycles = (
        train.groupby("unit_id")["cycle"]
        .max()
    )

    test_cycles = (
        test.groupby("unit_id")["cycle"]
        .max()
    )

    print("\nCYCLE INFORMATION")

    print(
        f"  Training min cycles : "
        f"{train_cycles.min()}"
    )

    print(
        f"  Training max cycles : "
        f"{train_cycles.max()}"
    )

    print(
        f"  Testing min cycles  : "
        f"{test_cycles.min()}"
    )

    print(
        f"  Testing max cycles  : "
        f"{test_cycles.max()}"
    )

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    print("\nMISSING VALUES")

    print(
        f"  Training missing values : "
        f"{train.isna().sum().sum()}"
    )

    print(
        f"  Testing missing values  : "
        f"{test.isna().sum().sum()}"
    )

    print(
        f"  RUL missing values      : "
        f"{rul.isna().sum().sum()}"
    )

    # ---------------------------------------------------------
    # Sample rows
    # ---------------------------------------------------------

    print("\nTRAINING SAMPLE")

    print(
        train.head(3).to_string(index=False)
    )

    print("\nRUL SAMPLE")

    print(
        rul.head(10).to_string(index=False)
    )

    # ---------------------------------------------------------
    # RUL statistics
    # ---------------------------------------------------------

    print("\nRUL STATISTICS")

    print(
        f"  Minimum : {rul['RUL'].min()}"
    )

    print(
        f"  Maximum : {rul['RUL'].max()}"
    )

    print(
        f"  Mean    : {rul['RUL'].mean():.2f}"
    )

    print(
        f"  Median  : {rul['RUL'].median():.2f}"
    )

    print("\n" + "=" * 70)
    print("C-MAPSS FD001 VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_validation()
