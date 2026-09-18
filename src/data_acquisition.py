from pathlib import Path
import json

from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ai4i"

DATASET_ID = 601


def acquire_ai4i():
    """
    Acquire the AI4I 2020 Predictive Maintenance Dataset
    from the UCI Machine Learning Repository.

    The original dataset is preserved without removing
    identifier columns. Modeling transformations will be
    performed later in the preprocessing stage.
    """

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching AI4I 2020 from UCI...")

    dataset = fetch_ucirepo(id=DATASET_ID)

    # Preserve the original dataset.
    original_data = dataset.data.original

    csv_path = RAW_DATA_DIR / "ai4i2020.csv"
    metadata_path = RAW_DATA_DIR / "metadata.json"

    original_data.to_csv(csv_path, index=False)

    metadata = {
        "dataset_name": "AI4I 2020 Predictive Maintenance Dataset",
        "source": "UCI Machine Learning Repository",
        "uci_dataset_id": DATASET_ID,
        "purpose": "Supervised predictive maintenance and machine failure prediction",
        "rows": int(original_data.shape[0]),
        "columns": int(original_data.shape[1]),
        "synthetic": True,
        "preserved_original": True,
        "source_url": (
            "https://archive.ics.uci.edu/dataset/601/"
            "ai4i%2B2020%2Bpredictive%2Bmaintenanc"
        ),
    }

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print(f"Saved dataset: {csv_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Shape: {original_data.shape}")

    return csv_path


if __name__ == "__main__":
    acquire_ai4i()