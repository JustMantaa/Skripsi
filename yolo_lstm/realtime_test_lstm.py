import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DEFAULT_MODEL_PATH = "output/lstm_model.pt"
DEFAULT_META_PATH = "output/lstm_meta_torch.json"
DEFAULT_LANDMARKER_PATH = "hand_landmarker.task"
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_FRAME = NUM_LANDMARKS * COORDS_PER_LANDMARK
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden1: int,
        hidden2: int,
        num_classes: int,
        dense1: int = 32,
        dense2: int = 16,
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
        self.fc1 = nn.Linear(lstm2_out, dense1)
        self.dropout3 = nn.Dropout(dropout_dense)
        self.fc_mid = nn.Linear(dense1, dense2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense2, num_classes)

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


def load_torch_runtime(model_path: Path):
    ckpt = torch.load(str(model_path), map_location="cpu")
    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint .pt tidak valid: 'model_state_dict' tidak ditemukan")

    classes = [str(c) for c in ckpt["classes"]]
    timesteps = int(ckpt["timesteps"])
    features_per_frame = int(ckpt["features_per_frame"])
    mean = np.asarray(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32)

    state_dict = ckpt["model_state_dict"]
    hidden1 = int(ckpt.get("hidden1", state_dict["lstm1.weight_ih_l0"].shape[0] // 4))
    hidden2 = int(ckpt.get("hidden2", state_dict["lstm2.weight_ih_l0"].shape[0] // 4))
    bidirectional = bool(ckpt.get("bidirectional", "lstm2.weight_ih_l0_reverse" in state_dict))
    dense1 = int(ckpt.get("dense1", state_dict["fc1.weight"].shape[0]))
    dense2 = int(ckpt.get("dense2", state_dict.get("fc_mid.weight", state_dict["fc1.weight"]).shape[0]))
    dropout_lstm = float(ckpt.get("dropout_lstm", 0.5))
    dropout_dense = float(ckpt.get("dropout_dense", 0.3))

    model = LSTMClassifier(
        input_size=features_per_frame,
        hidden1=hidden1,
        hidden2=hidden2,
        num_classes=len(classes),
        dense1=dense1,
        dense2=dense2,
        bidirectional=bidirectional,
        dropout_lstm=dropout_lstm,
        dropout_dense=dropout_dense,
    )

    # Backward compatibility for older checkpoint without fc_mid.
    if "fc_mid.weight" not in state_dict:
        model.fc_mid = nn.Identity()
        model.fc2 = nn.Linear(dense1, len(classes))

    model.load_state_dict(state_dict)
    model.eval()

    return {
        "backend": "torch",
        "model": model,
        "timesteps": timesteps,
        "features_per_frame": features_per_frame,
        "classes": classes,
        "mean": mean,
        "scale": scale,
    }


def load_tf_runtime(model_path: Path, meta_path: Path):
    meta = load_meta(meta_path)
    model = tf.keras.models.load_model(str(model_path))
    return {
        "backend": "tf",
        "model": model,
        "timesteps": int(meta["timesteps"]),
        "features_per_frame": int(meta["features_per_frame"]),
        "classes": meta["classes"],
        "mean": meta["mean"],
        "scale": meta["scale"],
    }


def predict_probs(runtime: dict, x_input: np.ndarray) -> np.ndarray:
    if runtime["backend"] == "tf":
        return runtime["model"].predict(x_input, verbose=0)[0]

    x_t = torch.from_numpy(x_input.astype(np.float32))
    with torch.no_grad():
        logits = runtime["model"](x_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def load_meta(meta_path: Path) -> dict:
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


def extract_keypoints(detector, frame_bgr: np.ndarray, timestamp_ms: int):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
    )
    result = detector.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        handedness = "Unknown"
        if result.handedness and len(result.handedness) > 0 and len(result.handedness[0]) > 0:
            handedness = str(result.handedness[0][0].category_name)

        keypoints = []
        for lm in hand:
            keypoints.extend([lm.x, lm.y, lm.z])
        return np.asarray(keypoints, dtype=np.float32), True, handedness

    return np.zeros(FEATURES_PER_FRAME, dtype=np.float32), False, "None"


def normalize_keypoint_sequence(sequence: np.ndarray) -> np.ndarray:
    t, f = sequence.shape
    expected = NUM_LANDMARKS * COORDS_PER_LANDMARK
    if f != expected:
        raise ValueError(f"Expected {expected} features/frame, got {f}")

    seq3 = sequence.reshape(t, NUM_LANDMARKS, COORDS_PER_LANDMARK).astype(np.float32)
    out = np.empty_like(seq3)
    wrist_ref = seq3[0, WRIST_INDEX : WRIST_INDEX + 1, :]

    for i in range(t):
        frame = seq3[i]
        centered = frame - wrist_ref

        scale = float(np.linalg.norm(frame[MIDDLE_MCP_INDEX] - frame[WRIST_INDEX]))
        if not np.isfinite(scale) or scale < 1e-6:
            distances = np.linalg.norm(centered[:, :2], axis=1)
            scale = float(np.max(distances)) if distances.size else 1.0
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        out[i] = centered / scale

    return out.reshape(t, f).astype(np.float32)


def apply_scaler(sequence: np.ndarray, mean_flat: np.ndarray, scale_flat: np.ndarray) -> np.ndarray:
    x_flat = sequence.reshape(1, -1).astype(np.float32)
    safe_scale = np.where(scale_flat == 0, 1.0, scale_flat).astype(np.float32)
    x_scaled = (x_flat - mean_flat) / safe_scale
    return x_scaled.reshape(1, sequence.shape[0], sequence.shape[1]).astype(np.float32)


def format_topk(classes, probs, top_k=3):
    k = max(1, min(top_k, len(classes)))
    top_idx = np.argsort(probs)[::-1][:k]
    return [(classes[int(i)], float(probs[int(i)])) for i in top_idx]


def estimate_wrist_motion(sequence: np.ndarray) -> float:
    """Rata-rata perpindahan wrist antar frame dalam koordinat mentah (x, y)."""
    if sequence.shape[0] < 2:
        return 0.0
    wrist_xy = sequence[:, 0:2]
    delta = np.diff(wrist_xy, axis=0)
    return float(np.mean(np.linalg.norm(delta, axis=1)))


def main():
    parser = argparse.ArgumentParser(description="Realtime test LSTM dari webcam")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path model .h5/.keras atau .pt")
    parser.add_argument("--meta", default=DEFAULT_META_PATH, help="Path metadata .json (dipakai untuk model TF)")
    parser.add_argument("--landmarker", default=DEFAULT_LANDMARKER_PATH, help="Path hand_landmarker.task")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Index kamera (default 1)")
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH, help="Lebar frame kamera")
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT, help="Tinggi frame kamera")
    parser.add_argument("--top-k", type=int, default=2, help="Jumlah top probabilitas")
    parser.add_argument("--pred-every", type=int, default=2, help="Prediksi setiap N frame")
    parser.add_argument(
        "--none-label",
        default="none",
        help="Nama kelas non-isyarat. Jika confidence rendah, prediksi dipaksa ke label ini bila tersedia.",
    )
    parser.add_argument(
        "--none-threshold",
        type=float,
        default=0.55,
        help="Jika confidence prediksi < threshold, hasil dipaksa ke kelas NONE (jika kelas NONE ada).",
    )
    parser.add_argument(
        "--d-min-prob",
        type=float,
        default=0.30,
        help="Probabilitas minimum kelas D agar bisa diprioritaskan saat gesture statis.",
    )
    parser.add_argument(
        "--static-motion-threshold",
        type=float,
        default=0.015,
        help="Ambang gerakan wrist rata-rata agar dianggap statis.",
    )
    parser.add_argument(
        "--vote-window",
        type=int,
        default=5,
        help="Ukuran jendela majority vote untuk stabilisasi label realtime.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    meta_path = Path(args.meta)
    landmarker_path = Path(args.landmarker)

    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not landmarker_path.exists():
        raise FileNotFoundError(f"Landmarker tidak ditemukan: {landmarker_path}")

    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        runtime = load_torch_runtime(model_path)
    else:
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata tidak ditemukan: {meta_path}")
        runtime = load_tf_runtime(model_path, meta_path)

    detector = create_detector(landmarker_path)

    timesteps = int(runtime["timesteps"])
    features_per_frame = int(runtime["features_per_frame"])
    classes = runtime["classes"]
    mean = runtime["mean"]
    scale = runtime["scale"]

    if features_per_frame != FEATURES_PER_FRAME:
        raise ValueError(
            f"Model mengharapkan {features_per_frame} fitur/frame, "
            f"sementara script realtime ini mengeluarkan {FEATURES_PER_FRAME}."
        )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka kamera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))

    sequence_buffer = deque(maxlen=timesteps)
    state = "WAIT_HAND"
    last_label = "-"
    last_conf = 0.0
    last_topk = []
    last_handedness = "None"
    last_motion = 0.0
    pred_history = deque(maxlen=max(1, args.vote_window))

    frame_count = 0
    start_time = time.time()

    print("Realtime test berjalan. Tekan 'q' untuk keluar.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame = frame
            # Mirror hanya untuk tampilan agar UX natural, bukan untuk input model.
            display_frame = cv2.flip(raw_frame, 1)

            frame_count += 1
            timestamp_ms = int((time.time() - start_time) * 1000)

            keypoints, hand_detected, handedness = extract_keypoints(
                detector,
                raw_frame,
                timestamp_ms,
            )
            last_handedness = handedness

            if state == "WAIT_HAND":
                if hand_detected:
                    pred_history.clear()
                    sequence_buffer.clear()
                    sequence_buffer.append(keypoints)
                    state = "COLLECTING"
            elif state == "COLLECTING":
                if hand_detected:
                    sequence_buffer.append(keypoints)
                    if len(sequence_buffer) >= timesteps and (frame_count % max(1, args.pred_every) == 0):
                        seq = np.asarray(sequence_buffer, dtype=np.float32)
                        motion_score = estimate_wrist_motion(seq)
                        last_motion = motion_score
                        seq = normalize_keypoint_sequence(seq)
                        x_input = apply_scaler(seq, mean, scale)
                        probs = predict_probs(runtime, x_input)
                        pred_idx = int(np.argmax(probs))
                        raw_label = classes[pred_idx]
                        raw_conf = float(probs[pred_idx])

                        pred_label = raw_label
                        pred_conf = raw_conf

                        # Heuristik khusus gesture statis D agar tidak sering jatuh ke NONE.
                        if "D" in classes:
                            d_idx = classes.index("D")
                            d_prob = float(probs[d_idx])
                            if motion_score <= float(args.static_motion_threshold) and d_prob >= float(args.d_min_prob):
                                pred_label = "D"
                                pred_conf = d_prob

                        # Fallback ke kelas NONE untuk prediksi confidence rendah.
                        none_label = str(args.none_label)
                        if none_label in classes and pred_label != "D" and pred_conf < float(args.none_threshold):
                            forced_idx = classes.index(none_label)
                            pred_label = none_label
                            pred_conf = float(probs[forced_idx])

                        pred_history.append(pred_label)
                        # Majority vote untuk mengurangi flicker antar kelas.
                        values, counts = np.unique(np.array(pred_history, dtype=object), return_counts=True)
                        last_label = str(values[int(np.argmax(counts))])
                        last_conf = pred_conf
                        last_topk = format_topk(classes, probs, top_k=args.top_k)

                        # One-shot inference per 30 valid frames, then wait for next gesture.
                        sequence_buffer.clear()
                        state = "WAIT_HAND"
                else:
                    # If hand is lost during collection, restart collection.
                    sequence_buffer.clear()
                    state = "WAIT_HAND"

            # Overlay UI
            cv2.putText(
                display_frame,
                f"Prediksi: {last_label} ({last_conf:.3f})",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Buffer: {len(sequence_buffer)}/{timesteps}",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"State: {state}",
                (15, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"Hand: {last_handedness}",
                (15, 126),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 255, 200),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                f"VoteWin: {len(pred_history)}/{max(1, args.vote_window)} Thr:{args.none_threshold:.2f}",
                (15, 154),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 220, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display_frame,
                f"Motion: {last_motion:.4f}",
                (15, 182),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 180),
                2,
                cv2.LINE_AA,
            )

            y = 210
            for lbl, prob in last_topk:
                cv2.putText(
                    display_frame,
                    f"{lbl}: {prob:.3f}",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                y += 28

            cv2.imshow("Realtime LSTM SIBI (J/Z)", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
