from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cmapss"

X = np.load(DATA_DIR / "X_train.npy")
y = np.load(DATA_DIR / "y_train.npy")
units = np.load(DATA_DIR / "train_units.npy")


def main():

    print("\n" + "=" * 70)
    print("FORGESHIELD | ENGINE-WISE TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    unique_units = np.unique(units)

    print(f"\nTotal training engines: {len(unique_units)}")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(
            X,
            y,
            groups=units,
        )
    )

    X_train = X[train_idx]
    y_train = y[train_idx]
    units_train = units[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]
    units_val = units[val_idx]

    print("\nSPLIT RESULTS")

    print(f"  Training windows   : {len(X_train)}")
    print(f"  Validation windows : {len(X_val)}")

    print(f"  Training engines   : {len(np.unique(units_train))}")
    print(f"  Validation engines : {len(np.unique(units_val))}")

    overlap = set(units_train) & set(units_val)

    print(f"\nENGINE OVERLAP")
    print(f"  Shared engines: {len(overlap)}")

    if len(overlap) != 0:
        raise ValueError(
            "Engine leakage detected!"
        )

    np.save(DATA_DIR / "X_train_split.npy", X_train)
    np.save(DATA_DIR / "y_train_split.npy", y_train)
    np.save(DATA_DIR / "train_split_units.npy", units_train)

    np.save(DATA_DIR / "X_val.npy", X_val)
    np.save(DATA_DIR / "y_val.npy", y_val)
    np.save(DATA_DIR / "val_units.npy", units_val)

    print("\nSaved files:")

    print("  ✓ X_train_split.npy")
    print("  ✓ y_train_split.npy")
    print("  ✓ train_split_units.npy")
    print("  ✓ X_val.npy")
    print("  ✓ y_val.npy")
    print("  ✓ val_units.npy")

    print("\n" + "=" * 70)
    print("ENGINE-WISE SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
