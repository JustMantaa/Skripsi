import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


def apply_scaler(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    n, t, f = X.shape
    flat = X.reshape(n, t * f)
    out = ((flat - mean) / scale).astype(np.float32)
    return out.reshape(n, t, f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LSTM model")
    parser.add_argument("--data", default=os.path.join("output", "test.csv"))
    parser.add_argument("--model", default=os.path.join("output", "lstm_model.h5"))
    parser.add_argument("--meta", default=os.path.join("output", "lstm_meta.json"))
    parser.add_argument("--report-out", default=os.path.join("output", "evaluation_report.txt"))
    args = parser.parse_args()

    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta["classes"]
    mean = np.asarray(meta["scaler"]["mean"], dtype=np.float32)
    scale = np.asarray(meta["scaler"]["scale"], dtype=np.float32)

    df = pd.read_csv(args.data)
    if "label" not in df.columns:
        raise ValueError("Test CSV must contain a 'label' column")

    feature_cols = [c for c in df.columns if c != "label"]
    X_test_flat = df[feature_cols].to_numpy(dtype=np.float32)
    if X_test_flat.shape[1] % 63 != 0:
        raise ValueError("Feature count must be divisible by 63 (21 landmarks * xyz)")
    timesteps = X_test_flat.shape[1] // 63
    X_test = X_test_flat.reshape(-1, timesteps, 63)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_test = df["label"].astype(str).map(class_to_idx).to_numpy()
    if np.isnan(y_test).any():
        raise ValueError("Test CSV contains labels not present in training metadata")
    y_test = y_test.astype(np.int32)

    X_test_scaled = apply_scaler(X_test, mean, scale)

    model = tf.keras.models.load_model(args.model)
    probs = model.predict(X_test_scaled, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    report_dir = os.path.dirname(args.report_out)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)
    print(f"\nSaved report: {args.report_out}")


if __name__ == "__main__":
    main()
