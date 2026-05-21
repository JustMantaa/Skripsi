import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parents[1]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden1: int,
        hidden2: int,
        num_classes: int,
        bidirectional: bool = True,
        dropout_lstm: float = 0.5,
        dropout_dense: float = 0.3,
    ):
        super().__init__()
        self.bidirectional = bool(bidirectional)
        lstm2_out = hidden2 * (2 if self.bidirectional else 1)
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_lstm)
        self.lstm2 = nn.LSTM(
            input_size=hidden1,
            hidden_size=hidden2,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.dropout2 = nn.Dropout(dropout_lstm)
        self.fc1 = nn.Linear(lstm2_out, 32)
        self.dropout3 = nn.Dropout(dropout_dense)
        self.fc_mid = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    @staticmethod
    def _last_valid_timestep(x: torch.Tensor, seq_out: torch.Tensor) -> torch.Tensor:
        # Approximate Keras Masking(mask_value=0.0): take last non-zero timestep output.
        # x shape: [B, T, F], seq_out shape: [B, T, H]
        valid = (x.abs().sum(dim=2) > 0)
        lengths = valid.sum(dim=1)
        last_idx = torch.clamp(lengths - 1, min=0)
        batch_idx = torch.arange(seq_out.size(0), device=seq_out.device)
        return seq_out[batch_idx, last_idx, :]

    def forward(self, x):
        x_in = x
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self._last_valid_timestep(x_in, x)
        x = self.dropout2(x)
        x = self.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.relu(self.fc_mid(x))
        x = self.fc2(x)
        return x


def build_loaders(train_csv: str, val_csv: str, test_csv: str, batch_size: int):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        if "label" not in df.columns:
            raise ValueError(f"{name} CSV wajib punya kolom 'label'")

    feature_cols = [c for c in train_df.columns if c != "label"]
    val_feature_cols = [c for c in val_df.columns if c != "label"]
    test_feature_cols = [c for c in test_df.columns if c != "label"]
    if feature_cols != val_feature_cols or feature_cols != test_feature_cols:
        raise ValueError("Kolom fitur train/val/test harus sama")

    classes = sorted(train_df["label"].astype(str).unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    y_train = train_df["label"].astype(str).map(class_to_idx).to_numpy(dtype=np.int64)
    y_val = val_df["label"].astype(str).map(class_to_idx).to_numpy(dtype=np.int64)
    y_test = test_df["label"].astype(str).map(class_to_idx).to_numpy(dtype=np.int64)

    if np.isnan(y_val).any():
        raise ValueError("Ada label di val yang tidak ada di train")
    if np.isnan(y_test).any():
        raise ValueError("Ada label di test yang tidak ada di train")

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    if X_train.shape[1] % 63 != 0:
        raise ValueError("Jumlah fitur harus habis dibagi 63")

    timesteps = X_train.shape[1] // 63
    X_train = X_train.reshape(-1, timesteps, 63)
    X_val = X_val.reshape(-1, timesteps, 63)
    X_test = X_test.reshape(-1, timesteps, 63)

    scaler = StandardScaler()
    n_train, t, f = X_train.shape
    scaler.fit(X_train.reshape(n_train, t * f))

    def tx(x):
        n = x.shape[0]
        z = scaler.transform(x.reshape(n, t * f)).astype(np.float32)
        return z.reshape(n, t, f)

    X_train = tx(X_train)
    X_val = tx(X_val)
    X_test = tx(X_test)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, classes, timesteps, scaler


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_count += xb.size(0)

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser(description="Train LSTM with PyTorch and save .pt")
    parser.add_argument("--train", default=str(BASE_DIR / "output" / "train_aug.csv"))
    parser.add_argument("--val", default=str(BASE_DIR / "output" / "val.csv"))
    parser.add_argument("--test", default=str(BASE_DIR / "output" / "test.csv"))
    parser.add_argument("--model-out", default=str(BASE_DIR / "output" / "lstm_model.pt"))
    parser.add_argument("--meta-out", default=str(BASE_DIR / "output" / "lstm_meta_torch.json"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--bidirectional", dest="bidirectional", action="store_true", help="Aktifkan BiLSTM pada layer LSTM kedua")
    parser.add_argument("--no-bidirectional", dest="bidirectional", action="store_false", help="Nonaktifkan BiLSTM pada layer LSTM kedua")
    parser.set_defaults(bidirectional=True)
    parser.add_argument("--dropout-lstm", type=float, default=0.3)
    parser.add_argument("--dropout-dense", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    if args.hidden1 <= 0 or args.hidden2 <= 0:
        raise ValueError("--hidden1 dan --hidden2 harus > 0")
    if not 0.0 <= args.dropout_lstm < 1.0:
        raise ValueError("--dropout-lstm harus di rentang [0, 1)")
    if not 0.0 <= args.dropout_dense < 1.0:
        raise ValueError("--dropout-dense harus di rentang [0, 1)")

    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)
    model_out_path = Path(args.model_out)
    meta_out_path = Path(args.meta_out)

    if not train_path.is_absolute():
        train_path = BASE_DIR / train_path
    if not val_path.is_absolute():
        val_path = BASE_DIR / val_path
    if not test_path.is_absolute():
        test_path = BASE_DIR / test_path
    if not model_out_path.is_absolute():
        model_out_path = BASE_DIR / model_out_path
    if not meta_out_path.is_absolute():
        meta_out_path = BASE_DIR / meta_out_path

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader, classes, timesteps, scaler = build_loaders(
        str(train_path), str(val_path), str(test_path), args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(
        input_size=63,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        num_classes=len(classes),
        bidirectional=args.bidirectional,
        dropout_lstm=args.dropout_lstm,
        dropout_dense=args.dropout_dense,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_loss = float("inf")
    wait = 0

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_count += xb.size(0)

        train_loss = total_loss / max(1, total_count)
        train_acc = total_correct / max(1, total_count)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        hist["train_loss"].append(float(train_loss))
        hist["train_acc"].append(float(train_acc))
        hist["val_loss"].append(float(val_loss))
        hist["val_acc"].append(float(val_acc))

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping di epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "timesteps": int(timesteps),
        "features_per_frame": 63,
        "hidden1": int(args.hidden1),
        "hidden2": int(args.hidden2),
        "bidirectional": bool(args.bidirectional),
        "dense1": 32,
        "dense2": 16,
        "dropout_lstm": float(args.dropout_lstm),
        "dropout_dense": float(args.dropout_dense),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "weight_decay": float(args.weight_decay),
        "data_split": "train/val/test from CSV files",
    }
    torch.save(checkpoint, model_out_path)

    meta = {
        "framework": "pytorch",
        "classes": classes,
        "timesteps": int(timesteps),
        "features_per_frame": 63,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "history": hist,
    }
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model .pt: {model_out_path}")
    print(f"Saved meta: {meta_out_path}")
    print(f"test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
