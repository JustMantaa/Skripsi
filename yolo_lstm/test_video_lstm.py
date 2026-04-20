import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DEFAULT_MODEL_PATH = "output/lstm_model.h5"
DEFAULT_META_PATH = "output/lstm_meta.json"
DEFAULT_LANDMARKER_PATH = "hand_landmarker.task"
DEFAULT_VIDEO_PATH = "dataset/jtest2.mp4"
DEFAULT_EVAL_MODE = "single"
DEFAULT_DATASET_DIR = "dataset"
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9


def load_meta(meta_path: Path):
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    required = ["timesteps", "features_per_frame", "classes", "scaler"]
    for key in required:
        if key not in meta:
            raise ValueError(f"Metadata tidak lengkap, key '{key}' tidak ditemukan.")

    scaler = meta["scaler"]
    if "mean" not in scaler or "scale" not in scaler:
        raise ValueError("Metadata scaler harus memiliki 'mean' dan 'scale'.")

    timesteps = int(meta["timesteps"])
    features_per_frame = int(meta["features_per_frame"])
    classes = [str(c) for c in meta["classes"]]
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)

    expected_flat = timesteps * features_per_frame
    if mean.shape[0] != expected_flat or scale.shape[0] != expected_flat:
        raise ValueError(
            "Ukuran scaler tidak sesuai dengan timesteps x features_per_frame "
            f"({expected_flat})."
        )

    return {
        "timesteps": timesteps,
        "features_per_frame": features_per_frame,
        "classes": classes,
        "mean": mean,
        "scale": scale,
    }


def create_detector(landmarker_path: Path):
    base_options = python.BaseOptions(model_asset_path=str(landmarker_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_keypoints(detector, frame_bgr, features_per_frame: int, timestamp_ms: int):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
    )
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        keypoints = []
        for lm in hand:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.asarray(keypoints, dtype=np.float32)

    return np.zeros(features_per_frame, dtype=np.float32)


def video_to_sequence(video_path: Path, detector, timesteps: int, features_per_frame: int):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(
            extract_keypoints(
                detector,
                frame,
                features_per_frame,
                timestamp_ms=frame_idx * 33,
            )
        )
        frame_idx += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"Video tidak bisa dibaca atau kosong: {video_path}")

    frames = np.asarray(frames, dtype=np.float32)

    # Samakan panjang sequence agar konsisten dengan training.
    if len(frames) < timesteps:
        pad = np.zeros((timesteps - len(frames), features_per_frame), dtype=np.float32)
        frames = np.vstack((frames, pad))
    elif len(frames) > timesteps:
        idx = np.linspace(0, len(frames) - 1, timesteps).astype(int)
        frames = frames[idx]

    return frames


def apply_scaler(sequence, mean_flat, scale_flat):
    x_flat = sequence.reshape(1, -1).astype(np.float32)

    # Hindari pembagian 0 jika ada fitur konstan saat training.
    safe_scale = np.where(scale_flat == 0, 1.0, scale_flat).astype(np.float32)
    x_scaled = (x_flat - mean_flat) / safe_scale

    return x_scaled.reshape(1, sequence.shape[0], sequence.shape[1]).astype(np.float32)


def normalize_keypoint_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Match the training-time keypoint normalization:
    1) Center each frame by wrist landmark.
    2) Scale each frame by wrist->middle_mcp distance.
    """
    t, f = sequence.shape
    expected = NUM_LANDMARKS * COORDS_PER_LANDMARK
    if f != expected:
        raise ValueError(f"Expected {expected} features/frame, got {f}")

    seq3 = sequence.reshape(t, NUM_LANDMARKS, COORDS_PER_LANDMARK).astype(np.float32)
    out = np.empty_like(seq3)

    for i in range(t):
        frame = seq3[i]
        wrist = frame[WRIST_INDEX : WRIST_INDEX + 1, :]
        centered = frame - wrist

        scale = float(np.linalg.norm(frame[MIDDLE_MCP_INDEX] - frame[WRIST_INDEX]))
        if not np.isfinite(scale) or scale < 1e-6:
            distances = np.linalg.norm(centered[:, :2], axis=1)
            scale = float(np.max(distances)) if distances.size else 1.0
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        out[i] = centered / scale

    return out.reshape(t, f).astype(np.float32)


def predict_video(
    video_path: Path,
    model_path: Path,
    meta_path: Path,
    landmarker_path: Path,
    top_k: int,
):
    if not video_path.exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {meta_path}")
    if not landmarker_path.exists():
        raise FileNotFoundError(f"Model landmarker tidak ditemukan: {landmarker_path}")

    meta = load_meta(meta_path)
    detector = create_detector(landmarker_path)

    sequence = video_to_sequence(
        video_path=video_path,
        detector=detector,
        timesteps=meta["timesteps"],
        features_per_frame=meta["features_per_frame"],
    )
    sequence = normalize_keypoint_sequence(sequence)
    x_input = apply_scaler(sequence, meta["mean"], meta["scale"])

    model = tf.keras.models.load_model(str(model_path))
    probs = model.predict(x_input, verbose=0)[0]

    classes = meta["classes"]
    if len(classes) != len(probs):
        raise ValueError(
            "Jumlah kelas di metadata tidak sama dengan output model "
            f"({len(classes)} vs {len(probs)})."
        )

    pred_idx = int(np.argmax(probs))
    pred_label = classes[pred_idx]
    pred_conf = float(probs[pred_idx])

    k = max(1, min(top_k, len(classes)))
    top_idx = np.argsort(probs)[::-1][:k]
    top_results = [(classes[int(i)], float(probs[int(i)])) for i in top_idx]

    return pred_label, pred_conf, top_results


def evaluate_dataset(
    dataset_dir: Path,
    model_path: Path,
    meta_path: Path,
    landmarker_path: Path,
):
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {meta_path}")
    if not landmarker_path.exists():
        raise FileNotFoundError(f"Model landmarker tidak ditemukan: {landmarker_path}")

    meta = load_meta(meta_path)
    classes = meta["classes"]
    class_to_idx = {label: i for i, label in enumerate(classes)}

    detector = create_detector(landmarker_path)
    model = tf.keras.models.load_model(str(model_path))

    video_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    y_true = []
    y_pred = []

    for label in classes:
        label_dir = dataset_dir / label
        if not label_dir.exists() or not label_dir.is_dir():
            continue

        files = sorted([p for p in label_dir.iterdir() if p.suffix.lower() in video_ext])
        for video_path in files:
            sequence = video_to_sequence(
                video_path=video_path,
                detector=detector,
                timesteps=meta["timesteps"],
                features_per_frame=meta["features_per_frame"],
            )
            sequence = normalize_keypoint_sequence(sequence)
            x_input = apply_scaler(sequence, meta["mean"], meta["scale"])
            probs = model.predict(x_input, verbose=0)[0]
            pred_idx = int(np.argmax(probs))

            y_true.append(class_to_idx[label])
            y_pred.append(pred_idx)

    if not y_true:
        raise ValueError(
            "Tidak ada video evaluasi yang diproses. Pastikan folder dataset berisi video "
            "dengan struktur dataset/<label>/*.mp4"
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
    acc = accuracy_score(y_true, y_pred)

    return {
        "total_y_actual": int(len(y_true)),
        "total_y_pred": int(len(y_pred)),
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def main():
    parser = argparse.ArgumentParser(description="Test/evaluasi model LSTM")
    parser.add_argument(
        "--mode",
        choices=["single", "dataset"],
        default=DEFAULT_EVAL_MODE,
        help="single: prediksi 1 video, dataset: evaluasi metrik dataset",
    )
    parser.add_argument(
        "--video",
        default=DEFAULT_VIDEO_PATH,
        help="Path video yang ingin diprediksi",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Folder dataset dengan struktur dataset/<label>/*.mp4",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path file model (.h5/.keras)")
    parser.add_argument("--meta", default=DEFAULT_META_PATH, help="Path file metadata .json")
    parser.add_argument(
        "--landmarker",
        default=DEFAULT_LANDMARKER_PATH,
        help="Path file hand_landmarker.task",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Jumlah top prediksi")
    args = parser.parse_args()

    if args.mode == "single":
        video_path = Path(args.video)
        if not video_path.exists():
            dataset_root = Path(args.dataset_dir)
            video_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
            candidates = sorted(
                [p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in video_ext]
            )
            if candidates:
                video_path = candidates[0]

        pred_label, pred_conf, top_results = predict_video(
            video_path=video_path,
            model_path=Path(args.model),
            meta_path=Path(args.meta),
            landmarker_path=Path(args.landmarker),
            top_k=args.top_k,
        )

        print("Hasil Prediksi")
        print(f"Label: {pred_label}")
        print(f"Confidence: {pred_conf:.4f}")
        print("Top Probabilities:")
        for label, prob in top_results:
            print(f"- {label}: {prob:.4f}")
    else:
        metrics = evaluate_dataset(
            dataset_dir=Path(args.dataset_dir),
            model_path=Path(args.model),
            meta_path=Path(args.meta),
            landmarker_path=Path(args.landmarker),
        )

        print("Hasil Evaluasi Dataset")
        print(f"Total y_actual: {metrics['total_y_actual']}")
        print(f"Total y_pred: {metrics['total_y_pred']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])
        print("Classification Report:")
        print(metrics["classification_report"])


if __name__ == "__main__":
    main()
