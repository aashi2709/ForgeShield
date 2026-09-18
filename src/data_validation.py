from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AI4I_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i" / "ai4i2020.csv"


def validate_ai4i(path: Path) -> None:
    """Run basic data-quality validation on the AI4I dataset."""

    print("\n" + "=" * 60)
    print("FORGESHIELD | AI4I DATA VALIDATION")
    print("=" * 60)

    df = pd.read_csv(path)

    print("\n[1] Dataset shape")
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")

    print("\n[2] Column information")
    print(df.dtypes.to_string())

    print("\n[3] Missing values")
    missing = df.isna().sum()
    print(missing.to_string())

    print(f"\nTotal missing values: {missing.sum()}")

    print("\n[4] Duplicate rows")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n[5] Unique values")
    for column in df.columns:
        print(f"{column}: {df[column].nunique()}")

    print("\n[6] Machine failure distribution")
    print(df["Machine failure"].value_counts().sort_index().to_string())

    print("\n[7] Failure-mode distribution")
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]

    for column in failure_modes:
        print(f"\n{column}")
        print(df[column].value_counts().sort_index().to_string())

    print("\n[8] Numerical summary")
    print(df.describe().T.to_string())

    print("\n[9] Candidate identifier columns")
    print("UID")
    print("Product ID")

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    validate_ai4i(AI4I_PATH)
