import pandas as pd
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cmapss"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "cmapss"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "cmapss"
)

WINDOW_SIZE = 30
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5
RANDOM_SEED = 42


class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
        latent_size=32,
    ):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.latent = nn.Linear(
            hidden_size,
            latent_size,
        )

        self.decoder_input = nn.Linear(
            latent_size,
            hidden_size,
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.output_layer = nn.Linear(
            hidden_size,
            input_size,
        )

    def forward(self, x):

        # Encoder
        _, (hidden, _) = self.encoder(x)

        # Last encoder hidden state
        encoded = hidden[-1]

        # Latent representation
        latent = self.latent(encoded)

        # Expand latent representation
        decoder_context = self.decoder_input(
            latent
        )

        decoder_context = decoder_context.unsqueeze(
            1
        )

        decoder_context = decoder_context.repeat(
            1,
            x.size(1),
            1,
        )

        # Decoder
        decoded, _ = self.decoder(
            decoder_context
        )

        reconstruction = self.output_layer(
            decoded
        )

        return reconstruction


def set_seed():

    np.random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


def load_data():

    X_train = np.load(
        DATA_DIR / "X_train.npy"
    )

    X_val = np.load(
        DATA_DIR / "X_val.npy"
    )

    X_test = np.load(
        DATA_DIR / "X_test.npy"
    )

    train_units = np.load(
        DATA_DIR / "train_units.npy"
    )

    val_units = np.load(
        DATA_DIR / "val_units.npy"
    )

    test_units = np.load(
        DATA_DIR / "test_units.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train.npy"
    )

    y_val = np.load(
        DATA_DIR / "y_val.npy"
    )

    y_test = np.load(
        DATA_DIR / "y_test.npy"
    )

    return (
        X_train,
        X_val,
        X_test,
        train_units,
        val_units,
        test_units,
        y_train,
        y_val,
        y_test,
    )


def create_healthy_mask(
    X,
    y_rul,
    percentile=70,
):
    """
    Define healthy training windows using RUL.

    Windows whose RUL is above the 70th percentile
    are treated as predominantly healthy.

    This label is used ONLY to construct the
    autoencoder's healthy training set.
    """

    threshold = np.percentile(
        y_rul,
        percentile,
    )

    mask = (
        y_rul >= threshold
    )

    return mask, threshold


def reconstruction_errors(
    model,
    X,
    device,
):

    model.eval()

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        )
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    errors = []

    with torch.no_grad():

        for (batch,) in loader:

            batch = batch.to(device)

            reconstruction = model(
                batch
            )

            error = torch.mean(
                (batch - reconstruction) ** 2,
                dim=(1, 2),
            )

            errors.append(
                error.cpu().numpy()
            )

    return np.concatenate(
        errors
    )


def calculate_metrics(
    errors,
    y_true,
    threshold,
):

    anomaly_predictions = (
        errors > threshold
    ).astype(int)

    y_true = np.asarray(
        y_true
    ).astype(int)

    tp = np.sum(
        (anomaly_predictions == 1)
        & (y_true == 1)
    )

    fp = np.sum(
        (anomaly_predictions == 1)
        & (y_true == 0)
    )

    fn = np.sum(
        (anomaly_predictions == 0)
        & (y_true == 1)
    )

    tn = np.sum(
        (anomaly_predictions == 0)
        & (y_true == 0)
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "anomalies": int(
            anomaly_predictions.sum()
        ),
    }


def find_threshold(
    healthy_errors,
    val_errors,
    val_labels,
):

    # Threshold candidates are based on
    # reconstruction-error distribution
    # of healthy training windows.

    percentiles = np.arange(
        90,
        100,
        0.5,
    )

    candidates = np.percentile(
        healthy_errors,
        percentiles,
    )

    results = []

    for threshold in candidates:

        metrics = calculate_metrics(
            val_errors,
            val_labels,
            threshold,
        )

        results.append(
            metrics
        )

    results = sorted(
        results,
        key=lambda x: x["f1"],
        reverse=True,
    )

    return results


def plot_training_loss(
    train_losses,
    val_losses,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        train_losses,
        label="Training Loss",
    )

    plt.plot(
        val_losses,
        label="Validation Loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Reconstruction MSE"
    )

    plt.title(
        "LSTM Autoencoder Training"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "lstm_autoencoder_loss.png",
        dpi=300,
    )

    plt.close()


def plot_error_distribution(
    normal_errors,
    anomaly_errors,
    threshold,
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.hist(
        normal_errors,
        bins=60,
        alpha=0.65,
        label="Normal",
    )

    plt.hist(
        anomaly_errors,
        bins=60,
        alpha=0.65,
        label="Known Failure",
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label="Anomaly Threshold",
    )

    plt.xlabel(
        "Reconstruction Error"
    )

    plt.ylabel(
        "Number of Windows"
    )

    plt.title(
        "LSTM Autoencoder Reconstruction Error"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "lstm_autoencoder_error_distribution.png",
        dpi=300,
    )

    plt.close()


def plot_test_errors(
    errors,
    labels,
    threshold,
):

    plt.figure(
        figsize=(12, 6)
    )

    indices = np.arange(
        len(errors)
    )

    plt.plot(
        indices,
        errors,
        linewidth=0.8,
        label="Reconstruction Error",
    )

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=2,
        label="Anomaly Threshold",
    )

    failure_indices = (
        labels == 1
    )

    plt.scatter(
        indices[failure_indices],
        errors[failure_indices],
        s=10,
        label="Known Failure",
    )

    plt.xlabel(
        "Test Window"
    )

    plt.ylabel(
        "Reconstruction Error"
    )

    plt.title(
        "Test Reconstruction Error and Detected Anomalies"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "lstm_autoencoder_test_errors.png",
        dpi=300,
    )

    plt.close()


def main():

    print("\n" + "=" * 70)
    print(
        "FORGESHIELD | LSTM AUTOENCODER ANOMALY DETECTION"
    )
    print("=" * 70)

    set_seed()

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
    # Load C-MAPSS data
    # ---------------------------------------------------------

    print(
        "\nLoading C-MAPSS data..."
    )

    (
        X_train,
        X_val,
        X_test,
        train_units,
        val_units,
        test_units,
        y_train,
        y_val,
        y_test,
    ) = load_data()

    print(
        f"  X_train: {X_train.shape}"
    )

    print(
        f"  X_val  : {X_val.shape}"
    )

    print(
        f"  X_test : {X_test.shape}"
    )

    # ---------------------------------------------------------
    # Healthy training subset
    # ---------------------------------------------------------

    healthy_mask, healthy_rul_threshold = (
        create_healthy_mask(
            X_train,
            y_train,
            percentile=70,
        )
    )

    X_healthy = X_train[
        healthy_mask
    ]

    print(
        "\nHealthy-window selection:"
    )

    print(
        f"  RUL threshold: "
        f"{healthy_rul_threshold:.2f}"
    )

    print(
        f"  Healthy windows: "
        f"{len(X_healthy)}"
    )

    print(
        f"  Total windows: "
        f"{len(X_train)}"
    )

    print(
        f"  Healthy percentage: "
        f"{100 * len(X_healthy) / len(X_train):.2f}%"
    )

    # ---------------------------------------------------------
    # Data loaders
    # ---------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(
            X_healthy,
            dtype=torch.float32,
        )
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_val,
            dtype=torch.float32,
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    input_size = X_train.shape[2]

    model = LSTMAutoencoder(
        input_size=input_size,
        hidden_size=64,
        latent_size=32,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    print(
        "\nTraining LSTM Autoencoder..."
    )

    train_losses = []
    val_losses = []

    best_val_loss = float(
        "inf"
    )

    patience_counter = 0

    best_model_path = (
        MODEL_DIR
        / "lstm_autoencoder_best.pt"
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        train_loss = 0.0

        for (batch,) in train_loader:

            batch = batch.to(device)

            optimizer.zero_grad()

            reconstruction = model(
                batch
            )

            loss = criterion(
                reconstruction,
                batch,
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * batch.size(0)
            )

        train_loss /= len(
            train_loader.dataset
        )

        # Validation reconstruction loss
        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for (batch,) in val_loader:

                batch = batch.to(device)

                reconstruction = model(
                    batch
                )

                loss = criterion(
                    reconstruction,
                    batch,
                )

                validation_loss += (
                    loss.item()
                    * batch.size(0)
                )

        validation_loss /= len(
            val_loader.dataset
        )

        train_losses.append(
            train_loss
        )

        val_losses.append(
            validation_loss
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {validation_loss:.6f}"
        )

        if validation_loss < best_val_loss:

            best_val_loss = (
                validation_loss
            )

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),
                    "input_size":
                        input_size,
                    "hidden_size":
                        64,
                    "latent_size":
                        32,
                },
                best_model_path,
            )

        else:

            patience_counter += 1

            if (
                patience_counter
                >= PATIENCE
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

    # ---------------------------------------------------------
    # Load best model
    # ---------------------------------------------------------

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ---------------------------------------------------------
    # Training healthy errors
    # ---------------------------------------------------------

    print(
        "\nCalculating reconstruction errors..."
    )

    healthy_errors = (
        reconstruction_errors(
            model,
            X_healthy,
            device,
        )
    )

    # ---------------------------------------------------------
    # Validation errors
    # ---------------------------------------------------------

    val_errors = (
        reconstruction_errors(
            model,
            X_val,
            device,
        )
    )

    # Known failure definition:
    # RUL <= 30 is treated as degraded/failure-near.
    val_labels = (
        y_val <= 30
    ).astype(int)

    threshold_results = find_threshold(
        healthy_errors,
        val_errors,
        val_labels,
    )

    best_threshold_result = (
        threshold_results[0]
    )

    threshold = (
        best_threshold_result[
            "threshold"
        ]
    )

    print(
        "\nSelected anomaly threshold:"
    )

    print(
        f"  Threshold: {threshold:.6f}"
    )

    print(
        f"  Validation Precision: "
        f"{best_threshold_result['precision']:.4f}"
    )

    print(
        f"  Validation Recall: "
        f"{best_threshold_result['recall']:.4f}"
    )

    print(
        f"  Validation F1: "
        f"{best_threshold_result['f1']:.4f}"
    )

    # ---------------------------------------------------------
    # Test evaluation
    # ---------------------------------------------------------

    test_errors = (
        reconstruction_errors(
            model,
            X_test,
            device,
        )
    )

    test_labels = (
        y_test <= 30
    ).astype(int)

    test_metrics = calculate_metrics(
        test_errors,
        test_labels,
        threshold,
    )

    print(
        "\n" + "-" * 70
    )

    print(
        "HELD-OUT TEST EVALUATION"
    )

    print(
        f"  Threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"  Anomalies detected: "
        f"{test_metrics['anomalies']}"
    )

    print(
        f"  Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"  Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"  F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"  TP: {test_metrics['TP']}"
    )

    print(
        f"  FP: {test_metrics['FP']}"
    )

    print(
        f"  FN: {test_metrics['FN']}"
    )

    print(
        f"  TN: {test_metrics['TN']}"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    threshold_table = pd.DataFrame(
        threshold_results
    )

    threshold_table.to_csv(
        REPORT_DIR
        / "lstm_autoencoder_threshold_analysis.csv",
        index=False,
    )

    test_results = {
        "model":
            "LSTM Autoencoder",
        "threshold":
            threshold,
        "precision":
            test_metrics["precision"],
        "recall":
            test_metrics["recall"],
        "f1":
            test_metrics["f1"],
        "TP":
            test_metrics["TP"],
        "FP":
            test_metrics["FP"],
        "FN":
            test_metrics["FN"],
        "TN":
            test_metrics["TN"],
        "anomalies_detected":
            test_metrics["anomalies"],
    }

    pd.DataFrame(
        [test_results]
    ).to_csv(
        REPORT_DIR
        / "lstm_autoencoder_test_results.csv",
        index=False,
    )

    np.save(
        MODEL_DIR
        / "lstm_autoencoder_test_errors.npy",
        test_errors,
    )

    np.save(
        MODEL_DIR
        / "lstm_autoencoder_val_errors.npy",
        val_errors,
    )

    joblib.dump(
        {
            "threshold":
                threshold,
            "healthy_rul_percentile":
                70,
            "healthy_rul_threshold":
                healthy_rul_threshold,
            "window_size":
                WINDOW_SIZE,
        },
        MODEL_DIR
        / "lstm_autoencoder_metadata.joblib",
    )

    # ---------------------------------------------------------
    # Figures
    # ---------------------------------------------------------

    plot_training_loss(
        train_losses,
        val_losses,
    )

    normal_errors = test_errors[
        test_labels == 0
    ]

    anomaly_errors = test_errors[
        test_labels == 1
    ]

    plot_error_distribution(
        normal_errors,
        anomaly_errors,
        threshold,
    )

    plot_test_errors(
        test_errors,
        test_labels,
        threshold,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "LSTM AUTOENCODER COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSaved:"
    )

    print(
        "  ✓ Best model checkpoint"
    )

    print(
        "  ✓ Threshold analysis"
    )

    print(
        "  ✓ Held-out test results"
    )

    print(
        "  ✓ Reconstruction errors"
    )

    print(
        "  ✓ Training-loss figure"
    )

    print(
        "  ✓ Error-distribution figure"
    )

    print(
        "  ✓ Test-error figure"
    )


if __name__ == "__main__":
    main()

