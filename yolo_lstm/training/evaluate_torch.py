import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


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


def infer_bidirectional(state_dict: dict) -> bool:
    return "lstm2.weight_ih_l0_reverse" in state_dict


def infer_dense_sizes(state_dict: dict) -> tuple[int, int]:
    d1 = int(state_dict["fc1.weight"].shape[0])
    if "fc_mid.weight" in state_dict:
        d2 = int(state_dict["fc_mid.weight"].shape[0])
    else:
        d2 = d1
    return d1, d2


def plot_confusion_matrix(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(train_meta: dict, out_path: Path) -> bool:
    history = train_meta.get("history") if isinstance(train_meta, dict) else None
    if not isinstance(history, dict):
        return False

    train_loss = history.get("train_loss")
    val_loss = history.get("val_loss")
    train_acc = history.get("train_acc")
    val_acc = history.get("val_acc")

    has_loss = isinstance(train_loss, list) and len(train_loss) > 0
    has_acc = isinstance(train_acc, list) and len(train_acc) > 0
    if not has_loss and not has_acc:
        return False

    epochs = np.arange(1, (len(train_loss) if has_loss else len(train_acc)) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if has_loss:
        axes[0].plot(epochs, train_loss, label="Train Loss")
        if isinstance(val_loss, list) and len(val_loss) == len(train_loss):
            axes[0].plot(epochs, val_loss, label="Val Loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
    else:
        axes[0].axis("off")

    if has_acc:
        axes[1].plot(epochs, train_acc, label="Train Accuracy")
        if isinstance(val_acc, list) and len(val_acc) == len(train_acc):
            axes[1].plot(epochs, val_acc, label="Val Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
    else:
        axes[1].axis("off")

    fig.suptitle("Training History", fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch LSTM model (.pt)")
    parser.add_argument("--test", default=str(BASE_DIR / "output" / "test.csv"), help="Path to test CSV")
    parser.add_argument("--model", default=str(BASE_DIR / "output" / "lstm_model.pt"), help="Path to model .pt")
    parser.add_argument("--train-meta", default=str(BASE_DIR / "output" / "lstm_meta_torch.json"), help="Path to training metadata JSON (untuk plot history)")
    parser.add_argument("--meta-out", default=str(BASE_DIR / "output" / "evaluation_torch.json"), help="Path to save evaluation JSON")
    parser.add_argument("--report-out", default=str(BASE_DIR / "output" / "evaluation_torch_report.txt"), help="Path to save text report")
    parser.add_argument("--cm-out", default=str(BASE_DIR / "output" / "confusion_matrix_torch.png"), help="Path to save confusion matrix plot")
    parser.add_argument("--history-out", default=str(BASE_DIR / "output" / "training_history_torch.png"), help="Path to save training history plot")
    parser.add_argument(
        "--average",
        default="weighted",
        choices=["micro", "macro", "weighted"],
        help="Averaging strategy for Precision/Recall/F1 aggregate",
    )
    args = parser.parse_args()

    test_path = Path(args.test)
    model_path = Path(args.model)
    train_meta_path = Path(args.train_meta)
    meta_out_path = Path(args.meta_out)
    report_out_path = Path(args.report_out)
    cm_out_path = Path(args.cm_out)
    history_out_path = Path(args.history_out)

    if not test_path.is_absolute():
        test_path = BASE_DIR / test_path
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path
    if not train_meta_path.is_absolute():
        train_meta_path = BASE_DIR / train_meta_path
    if not meta_out_path.is_absolute():
        meta_out_path = BASE_DIR / meta_out_path
    if not report_out_path.is_absolute():
        report_out_path = BASE_DIR / report_out_path
    if not cm_out_path.is_absolute():
        cm_out_path = BASE_DIR / cm_out_path
    if not history_out_path.is_absolute():
        history_out_path = BASE_DIR / history_out_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    classes = checkpoint["classes"]
    timesteps = int(checkpoint["timesteps"])
    features_per_frame = int(checkpoint["features_per_frame"])
    scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
    scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)

    test_df = pd.read_csv(test_path)
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
    bidirectional = bool(checkpoint.get("bidirectional", infer_bidirectional(state_dict)))
    _dense1, dense2 = infer_dense_sizes(state_dict)
    dropout_lstm = float(checkpoint.get("dropout_lstm", 0.5))
    dropout_dense = float(checkpoint.get("dropout_dense", 0.3))
    model = LSTMClassifier(
        input_size=features_per_frame,
        hidden1=hidden1,
        hidden2=hidden2,
        num_classes=len(classes),
        bidirectional=bidirectional,
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        average=args.average,
        zero_division=0,
    )
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
    print("\nAggregate Metrics:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f} ({args.average})")
    print(f"Recall   : {recall:.4f} ({args.average})")
    print(f"F1-score : {f1:.4f} ({args.average})")
    print("\nClassification Report:")
    print(report)

    meta_out_path.parent.mkdir(parents=True, exist_ok=True)
    report_out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "average": args.average,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }
    with open(meta_out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(report_out_path, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nAggregate Metrics:\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f} ({args.average})\n")
        f.write(f"Recall   : {recall:.4f} ({args.average})\n")
        f.write(f"F1-score : {f1:.4f} ({args.average})\n")
        f.write("\n\nClassification Report:\n")
        f.write(report)

    plot_confusion_matrix(cm, classes, cm_out_path)

    has_history = False
    if train_meta_path.exists():
        with open(train_meta_path, "r", encoding="utf-8") as f:
            train_meta = json.load(f)
        has_history = plot_training_history(train_meta, history_out_path)

    print(f"Saved evaluation json: {meta_out_path}")
    print(f"Saved evaluation report: {report_out_path}")
    print(f"Saved confusion matrix plot: {cm_out_path}")
    if has_history:
        print(f"Saved training history plot: {history_out_path}")
    else:
        print("Training history tidak ditemukan, grafik history tidak dibuat.")


if __name__ == "__main__":
    main()
