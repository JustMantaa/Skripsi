import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden1: int,
        hidden2: int,
        num_classes: int,
        dropout_lstm: float = 0.5,
        dropout_dense: float = 0.3,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_lstm)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_lstm)
        self.fc1 = nn.Linear(hidden2, 32)
        self.dropout3 = nn.Dropout(dropout_dense)
        self.fc_mid = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    @staticmethod
    def _last_valid_timestep(x: torch.Tensor, seq_out: torch.Tensor) -> torch.Tensor:
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


def apply_scaler(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    n, t, f = X.shape
    flat = X.reshape(n, t * f)
    safe_scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    z = ((flat - mean) / safe_scale).astype(np.float32)
    return z.reshape(n, t, f)


def infer_hidden_sizes(state_dict: dict) -> tuple[int, int]:
    # weight_ih_l0 shape: (4*hidden1, input_size)
    h1 = state_dict["lstm1.weight_ih_l0"].shape[0] // 4
    # weight_ih_l0 shape in lstm2: (4*hidden2, hidden1)
    h2 = state_dict["lstm2.weight_ih_l0"].shape[0] // 4
    return int(h1), int(h2)


def infer_dense_sizes(state_dict: dict) -> tuple[int, int]:
    d1 = int(state_dict["fc1.weight"].shape[0])
    if "fc_mid.weight" in state_dict:
        d2 = int(state_dict["fc_mid.weight"].shape[0])
    else:
        d2 = d1
    return d1, d2


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch LSTM model (.pt)")
    parser.add_argument("--test", default=os.path.join("output", "test.csv"), help="Path to test CSV")
    parser.add_argument("--model", default=os.path.join("output", "lstm_model.pt"), help="Path to model .pt")
    parser.add_argument("--meta-out", default=os.path.join("output", "evaluation_torch.json"), help="Path to save evaluation JSON")
    parser.add_argument("--report-out", default=os.path.join("output", "evaluation_torch_report.txt"), help="Path to save text report")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model tidak ditemukan: {args.model}")

    checkpoint = torch.load(args.model, map_location="cpu")
    classes = checkpoint["classes"]
    timesteps = int(checkpoint["timesteps"])
    features_per_frame = int(checkpoint["features_per_frame"])
    scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)

    test_df = pd.read_csv(args.test)
    if "label" not in test_df.columns:
        raise ValueError("Test CSV wajib memiliki kolom 'label'")

    feature_cols = [c for c in test_df.columns if c != "label"]
    expected_features = timesteps * features_per_frame
    if len(feature_cols) != expected_features:
        raise ValueError(
            f"Jumlah kolom fitur tidak sesuai. Expected={expected_features}, got={len(feature_cols)}"
        )

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_s = test_df["label"].astype(str).map(class_to_idx)
    if y_true_s.isna().any():
        raise ValueError("Ada label test yang tidak ditemukan di classes model")

    y_true = y_true_s.to_numpy(dtype=np.int64)
    X = test_df[feature_cols].to_numpy(dtype=np.float32).reshape(-1, timesteps, features_per_frame)
    X = apply_scaler(X, scaler_mean, scaler_scale)

    state_dict = checkpoint["model_state_dict"]
    hidden1, hidden2 = infer_hidden_sizes(state_dict)
    _dense1, dense2 = infer_dense_sizes(state_dict)
    dropout_lstm = float(checkpoint.get("dropout_lstm", 0.5))
    dropout_dense = float(checkpoint.get("dropout_dense", 0.3))
    model = LSTMClassifier(
        input_size=features_per_frame,
        hidden1=hidden1,
        hidden2=hidden2,
        num_classes=len(classes),
        dropout_lstm=dropout_lstm,
        dropout_dense=dropout_dense,
    )

    # Backward compatibility: old checkpoints had no fc_mid.
    if "fc_mid.weight" not in state_dict:
        model.fc_mid = nn.Identity()
        model.fc2 = nn.Linear(32, len(classes))

    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        digits=4,
        zero_division=0,
    )

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"Accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)

    summary = {
        "accuracy": acc,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\n")

    print(f"Saved evaluation json: {args.meta_out}")
    print(f"Saved evaluation report: {args.report_out}")


if __name__ == "__main__":
    main()
