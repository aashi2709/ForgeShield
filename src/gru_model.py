from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "cmapss"
MODEL_DIR = PROJECT_ROOT / "models" / "cmapss"

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
PATIENCE = 5

SEED = 42


def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


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


def calculate_metrics(y_true, y_pred):

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    rmse = np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

    return mae, rmse


def main():

    set_seed()

    print("\n" + "=" * 70)
    print("FORGESHIELD | GRU RUL REGRESSOR")
    print("=" * 70)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    print("\nLoading datasets...")

    X_train = np.load(
        DATA_DIR / "X_train_split.npy"
    )

    y_train = np.load(
        DATA_DIR / "y_train_split.npy"
    )

    X_val = np.load(
        DATA_DIR / "X_val.npy"
    )

    y_val = np.load(
        DATA_DIR / "y_val.npy"
    )

    print(
        f"  X_train: {X_train.shape}"
    )

    print(
        f"  y_train: {y_train.shape}"
    )

    print(
        f"  X_val  : {X_val.shape}"
    )

    print(
        f"  y_val  : {y_val.shape}"
    )

    # ---------------------------------------------------------
    # Create datasets
    # ---------------------------------------------------------

    train_dataset = TensorDataset(
        torch.tensor(
            X_train,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_train,
            dtype=torch.float32,
        ),
    )

    val_dataset = TensorDataset(
        torch.tensor(
            X_val,
            dtype=torch.float32,
        ),
        torch.tensor(
            y_val,
            dtype=torch.float32,
        ),
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

    model = GRURegressor(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("\nMODEL")

    print(
        f"  Input features : {input_size}"
    )

    print(
        f"  Hidden size    : {HIDDEN_SIZE}"
    )

    print(
        f"  GRU layers     : {NUM_LAYERS}"
    )

    print(
        f"  Batch size     : {BATCH_SIZE}"
    )

    print(
        f"  Epochs         : {EPOCHS}"
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    best_val_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_model_path = (
        MODEL_DIR / "gru_best.pt"
    )

    print("\nTRAINING")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss = 0.0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = criterion(
                predictions,
                batch_y,
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * len(batch_X)
            )

        train_loss /= len(train_dataset)

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for batch_X, batch_y in val_loader:

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions = model(batch_X)

                loss = criterion(
                    predictions,
                    batch_y,
                )

                val_loss += (
                    loss.item()
                    * len(batch_X)
                )

        val_loss /= len(val_dataset)

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # -----------------------------------------------------
        # Early stopping
        # -----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                best_model_path,
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"\nEarly stopping at epoch {epoch}"
                )

                break

    # ---------------------------------------------------------
    # Load best model
    # ---------------------------------------------------------

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True,
        )
    )

    model.eval()

    # ---------------------------------------------------------
    # Validation predictions
    # ---------------------------------------------------------

    val_predictions = []

    with torch.no_grad():

        for batch_X, _ in val_loader:

            batch_X = batch_X.to(device)

            predictions = model(batch_X)

            val_predictions.extend(
                predictions.cpu().numpy()
            )

    val_predictions = np.asarray(
        val_predictions
    )

    val_mae, val_rmse = calculate_metrics(
        y_val,
        val_predictions,
    )

    print("\nVALIDATION RESULTS")

    print(
        f"  MAE  : {val_mae:.4f}"
    )

    print(
        f"  RMSE : {val_rmse:.4f}"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    np.save(
        MODEL_DIR / "gru_val_predictions.npy",
        val_predictions,
    )

    np.save(
        MODEL_DIR / "gru_train_loss.npy",
        np.asarray(
            history["train_loss"]
        ),
    )

    np.save(
        MODEL_DIR / "gru_val_loss.npy",
        np.asarray(
            history["val_loss"]
        ),
    )

    print("\nSaved files:")

    print(
        "  ✓ gru_best.pt"
    )

    print(
        "  ✓ gru_val_predictions.npy"
    )

    print(
        "  ✓ gru_train_loss.npy"
    )

    print(
        "  ✓ gru_val_loss.npy"
    )

    print("\n" + "=" * 70)
    print("GRU TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
