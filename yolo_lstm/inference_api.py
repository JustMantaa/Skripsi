import base64
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from flask_cors import CORS
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


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
        valid = x.abs().sum(dim=2) > 0
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


def infer_hidden_sizes(state_dict: dict) -> tuple[int, int]:
    h1 = state_dict["lstm1.weight_ih_l0"].shape[0] // 4
    h2 = state_dict["lstm2.weight_ih_l0"].shape[0] // 4
    return int(h1), int(h2)


def apply_scaler(x: np.ndarray, mean: np.ndarray | None, scale: np.ndarray | None) -> np.ndarray:
    if mean is None or scale is None:
        return x

    n, t, f = x.shape
    flat = x.reshape(n, t * f)
    safe_scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
    z = ((flat - mean) / safe_scale).astype(np.float32)
    return z.reshape(n, t, f)


def decode_data_url_to_bgr(data_url: str) -> np.ndarray:
    if "," in data_url:
        _, payload = data_url.split(",", 1)
    else:
        payload = data_url

    image_bytes = base64.b64decode(payload)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Gagal decode gambar dari request")
    return frame


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)

state_lock = threading.Lock()

threshold = 0.5
low_conf_limit = 5
yolo_stride = 2

mode = "YOLO"
low_conf_count = 0
sequence: list[np.ndarray] = []
frame_id = 0
yolo_frame_counter = 0
last_yolo_label = "Tidak terdeteksi"
last_yolo_conf = 0.0


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    global mode, low_conf_count, sequence, frame_id
    global yolo_frame_counter, last_yolo_label, last_yolo_conf

    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")

    if not image_data:
        return jsonify({"error": "Field 'image' wajib diisi"}), 400

    try:
        frame = decode_data_url_to_bgr(image_data)
    except Exception as exc:
        return jsonify({"error": f"Image tidak valid: {exc}"}), 400

    frame = cv2.flip(frame, 1)
    frame_id += 1

    response = {
        "label": "-",
        "confidence": 0.0,
        "mode": mode,
        "raw_mode": mode,
    }

    with state_lock:
        if mode == "YOLO":
            yolo_frame_counter += 1
            run_yolo = yolo_frame_counter % yolo_stride == 0
            if run_yolo:
                results = yolo(frame, verbose=False)[0]
            else:
                results = None

            if results is not None and len(results.boxes) > 0:
                conf = float(results.boxes.conf[0])
                cls = int(results.boxes.cls[0])
                label = yolo_labels[cls]

                last_yolo_label = label
                last_yolo_conf = conf

                response["label"] = label
                response["confidence"] = conf

                if conf < threshold:
                    low_conf_count += 1
                else:
                    low_conf_count = 0

                if low_conf_count >= low_conf_limit:
                    mode = "LSTM"
                    sequence = []
                    response["mode"] = "LSTM"
                    response["label"] = "Switch to LSTM"
                    response["confidence"] = conf
                else:
                    response["mode"] = "YOLO"
            else:
                if results is None:
                    response["label"] = last_yolo_label
                    response["confidence"] = last_yolo_conf
                    response["mode"] = "YOLO"
                else:
                    low_conf_count += 1
                    last_yolo_label = "Tidak terdeteksi"
                    last_yolo_conf = 0.0
                    response["label"] = "Tidak terdeteksi"
                    response["confidence"] = 0.0
                    response["mode"] = "YOLO"

        if mode == "LSTM" and len(sequence) == 0:
            response["label"] = "Collect 0/{}".format(seq_len)
            response["confidence"] = 0.0
            response["mode"] = "LSTM"

        if mode == "LSTM":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = hand_detector.detect_for_video(mp_image, frame_id)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                coords = []
                for lm in hand:
                    coords.extend([lm.x, lm.y, lm.z])

                coords_arr = np.array(coords, dtype=np.float32)

                wrist_x = coords_arr[0]
                wrist_y = coords_arr[1]
                for i in range(0, 63, 3):
                    coords_arr[i] -= wrist_x
                    coords_arr[i + 1] -= wrist_y

                sequence.append(coords_arr)

                response["label"] = f"Collect {len(sequence)}/{seq_len}"
                response["confidence"] = 0.0
                response["mode"] = "LSTM"
            else:
                response["label"] = "Tangan tidak terdeteksi"
                response["confidence"] = 0.0
                response["mode"] = "LSTM"

            if len(sequence) >= seq_len:
                x = np.array(sequence[:seq_len], dtype=np.float32)
                x = np.expand_dims(x, axis=0)
                x = apply_scaler(x, lstm_scaler_mean, lstm_scaler_scale)

                x_tensor = torch.tensor(x).to(device)

                with torch.no_grad():
                    logits = lstm(x_tensor)
                    probs = torch.softmax(logits, dim=1)
                    pred_class = int(torch.argmax(probs, dim=1).item())
                    conf = float(probs[0, pred_class].item())

                response["label"] = lstm_labels[pred_class]
                response["confidence"] = conf
                response["mode"] = "LSTM"

                mode = "YOLO"
                low_conf_count = 0
                sequence = []

    return jsonify(response)


if __name__ == "__main__":
    print("Memuat model YOLO + LSTM...")

    yolo = YOLO(str(BASE_DIR / "output" / "best.pt"))
    yolo_labels = yolo.names

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(str(BASE_DIR / "output" / "lstm_model.pt"), map_location=device)

    lstm_scaler_mean = None
    lstm_scaler_scale = None

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        features_per_frame = int(checkpoint.get("features_per_frame", 63))
        hidden1, hidden2 = infer_hidden_sizes(state_dict)

        lstm_labels = checkpoint.get("classes", ["J", "Z"])

        lstm = LSTMClassifier(
            input_size=features_per_frame,
            hidden1=hidden1,
            hidden2=hidden2,
            num_classes=len(lstm_labels),
            dropout_lstm=float(checkpoint.get("dropout_lstm", 0.5)),
            dropout_dense=float(checkpoint.get("dropout_dense", 0.3)),
        ).to(device)

        if "fc_mid.weight" not in state_dict:
            lstm.fc_mid = nn.Identity()
            lstm.fc2 = nn.Linear(32, len(lstm_labels)).to(device)

        lstm.load_state_dict(state_dict)

        if "scaler_mean" in checkpoint and "scaler_scale" in checkpoint:
            lstm_scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
            lstm_scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)

        seq_len = int(checkpoint.get("timesteps", 30))
    elif isinstance(checkpoint, nn.Module):
        lstm = checkpoint.to(device)
        lstm_labels = ["J", "Z"]
        seq_len = 30
    else:
        raise ValueError("Format output/lstm_model.pt tidak dikenali.")

    lstm.eval()

    base_options = python.BaseOptions(model_asset_path=str(BASE_DIR / "hand_landmarker.task"))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    hand_detector = vision.HandLandmarker.create_from_options(options)

    print("API siap di http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
