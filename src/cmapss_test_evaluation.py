from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cmapss"
MODEL_DIR = PROJECT_ROOT / "models" / "cmapss"
REPORT_DIR = PROJECT_ROOT / "reports" / "figures" / "cmapss"

SEED = 42


def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class LSTMRegressor(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        return self.fc(
            last_output
        ).squeeze(1)


class GRURegressor(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x):

        output, _ = self.gru(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        return self.fc(
            last_output
        ).squeeze(1)


class CNNLSTMRegressor(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            hidden_size,
            1,
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        x = self.conv(x)

        x = x.transpose(1, 2)

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        return self.fc(
            last_output
        ).squeeze(1)


def calculate_metrics(y_true, y_pred):

    errors = y_true - y_pred

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    return mae, rmse


def evaluate_model(
    model,
    X_test,
    y_test,
    device,
):

    model.eval()

    predictions = []

    batch_size = 256

    with torch.no_grad():

        for start in range(
            0,
            len(X_test),
            batch_size,
        ):

            end = start + batch_size

            batch_X = torch.tensor(
                X_test[start:end],
                dtype=torch.float32,
            ).to(device)

            batch_predictions = model(
                batch_X
            )

            predictions.extend(
                batch_predictions
                .cpu()
                .numpy()
            )

    predictions = np.asarray(
        predictions,
        dtype=np.float32,
    )

    mae, rmse = calculate_metrics(
        y_test,
        predictions,
    )

    return predictions, mae, rmse


def main():

    set_seed()

    print("\n" + "=" * 70)
    print(
        "FORGESHIELD | C-MAPSS FD001 TEST EVALUATION"
    )
    print("=" * 70)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # ---------------------------------------------------------
    # Load untouched test data
    # ---------------------------------------------------------

    print(
        "\nLoading untouched test data..."
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    test_units = np.load(
        DATA_DIR / "test_units.npy"
    )

    print(
        f"  X_test: {X_test.shape}"
    )

    print(
        f"  y_test: {y_test.shape}"
    )

    print(
        f"  Test engines: "
        f"{len(np.unique(test_units))}"
    )

    input_size = X_test.shape[2]

    # ---------------------------------------------------------
    # Model definitions
    # ---------------------------------------------------------

    models = {
        "LSTM": (
            LSTMRegressor(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
            ),
            MODEL_DIR / "lstm_best.pt",
        ),

        "GRU": (
            GRURegressor(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
            ),
            MODEL_DIR / "gru_best.pt",
        ),

        "CNN-LSTM": (
            CNNLSTMRegressor(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                dropout=0.2,
            ),
            MODEL_DIR / "cnn_lstm_best.pt",
        ),
    }

    results = []

    predictions_by_model = {}

    # ---------------------------------------------------------
    # Evaluate each model
    # ---------------------------------------------------------

    print(
        "\nTEST RESULTS"
    )

    for model_name, (
        model,
        model_path,
    ) in models.items():

        print(
            f"\nEvaluating {model_name}..."
        )

        model = model.to(device)

        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            state_dict
        )

        predictions, mae, rmse = (
            evaluate_model(
                model,
                X_test,
                y_test,
                device,
            )
        )

        predictions_by_model[
            model_name
        ] = predictions

        results.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
            }
        )

        print(
            f"  MAE  : {mae:.4f}"
        )

        print(
            f"  RMSE : {rmse:.4f}"
        )

    # ---------------------------------------------------------
    # Comparison table
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "MAE"
    ).reset_index(
        drop=True
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL C-MAPSS TEST COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df.to_csv(
        REPORT_DIR
        / "cmapss_test_model_comparison.csv",
        index=False,
    )

    for model_name, predictions in (
        predictions_by_model.items()
    ):

        filename = (
            model_name.lower()
            .replace("-", "_")
            + "_test_predictions.npy"
        )

        np.save(
            MODEL_DIR / filename,
            predictions,
        )

    # ---------------------------------------------------------
    # Per-engine analysis
    # ---------------------------------------------------------

    print(
        "\nPER-ENGINE ANALYSIS"
    )

    per_engine_rows = []

    for engine_id in np.unique(
        test_units
    ):

        mask = (
            test_units == engine_id
        )

        engine_true = y_test[mask]

        row = {
            "unit_id": int(engine_id),
            "num_windows": int(
                mask.sum()
            ),
        }

        for model_name, predictions in (
            predictions_by_model.items()
        ):

            engine_pred = predictions[mask]

            engine_mae = np.mean(
                np.abs(
                    engine_true
                    - engine_pred
                )
            )

            row[
                f"{model_name}_MAE"
            ] = engine_mae

        per_engine_rows.append(
            row
        )

    per_engine_df = pd.DataFrame(
        per_engine_rows
    )

    per_engine_df.to_csv(
        REPORT_DIR
        / "cmapss_per_engine_results.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Error statistics
    # ---------------------------------------------------------

    print(
        "\nERROR STATISTICS"
    )

    error_rows = []

    for model_name, predictions in (
        predictions_by_model.items()
    ):

        errors = (
            y_test - predictions
        )

        error_rows.append(
            {
                "model": model_name,
                "mean_error": np.mean(
                    errors
                ),
                "median_absolute_error": np.median(
                    np.abs(errors)
                ),
                "std_error": np.std(
                    errors
                ),
                "max_absolute_error": np.max(
                    np.abs(errors)
                ),
            }
        )

    error_df = pd.DataFrame(
        error_rows
    )

    print(
        error_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    error_df.to_csv(
        REPORT_DIR
        / "cmapss_error_statistics.csv",
        index=False,
    )

    print(
        "\nSaved files:"
    )

    print(
        "  ✓ cmapss_test_model_comparison.csv"
    )

    print(
        "  ✓ cmapss_per_engine_results.csv"
    )

    print(
        "  ✓ cmapss_error_statistics.csv"
    )

    print(
        "\nPrediction files saved in:"
    )

    print(
        f"  {MODEL_DIR}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "C-MAPSS TEST EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
