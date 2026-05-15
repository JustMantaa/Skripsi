import base64
import threading
import time
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


# =====================================================
# LSTM MODEL
# =====================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, num_classes, bidirectional=True, dropout_lstm=0.5, dropout_dense=0.3):
        super().__init__()
        self.bidirectional = bidirectional
        out_size = hidden2 * (2 if bidirectional else 1)

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_lstm)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True, bidirectional=bidirectional)
        self.dropout2 = nn.Dropout(dropout_lstm)
        self.fc1 = nn.Linear(out_size, 32)
        self.dropout3 = nn.Dropout(dropout_dense)
        self.fc_mid = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, num_classes)

    @staticmethod
    def _last_valid_timestep(x, seq_out):
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


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def infer_hidden_sizes(state_dict):
    h1 = state_dict["lstm1.weight_ih_l0"].shape[0] // 4
    h2 = state_dict["lstm2.weight_ih_l0"].shape[0] // 4
    return h1, h2


def infer_bidirectional(state_dict):
    return "lstm2.weight_ih_l0_reverse" in state_dict


def normalize_keypoint_sequence(sequence):
    t, f = sequence.shape
    seq3 = sequence.reshape(t, 21, 3).astype(np.float32)
    out = np.empty_like(seq3)
    wrist_ref = seq3[0, 0:1, :]

    for i in range(t):
        frame = seq3[i]
        centered = frame - wrist_ref
        scale = float(np.linalg.norm(frame[9] - frame[0]))
        if scale < 1e-6:
            scale = 1.0
        out[i] = centered / scale

    return out.reshape(t, f)


def apply_scaler(x, mean, scale):
    if mean is None:
        return x

    n, t, f = x.shape
    flat = x.reshape(n, t * f)
    scale_safe = np.where(scale == 0, 1.0, scale)
    flat = (flat - mean) / scale_safe
    return flat.reshape(n, t, f)


def decode_data_url(data_url):
    _, encoded = data_url.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def build_response(label="-", confidence=0.0, mode_name="LSTM", bbox=None):
    return {
        "label": label,
        "confidence": confidence,
        "mode": mode_name,
        "bbox": bbox,
    }


# =====================================================
# FLASK APP
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
CORS(app)
lock = threading.Lock()


# =====================================================
# CONFIG - SWITCHING PARAMETERS
# =====================================================
NO_DETECTION_LIMIT = 3
RESULT_CACHE_SECONDS = 3.0
LSTM_CONF_THRESHOLD = 0.70
LSTM_TIMEOUT = 5.0
HAND_MISSING_LIMIT = 5


# =====================================================
# GLOBAL STATE
# =====================================================
sequence = []
hand_missing_count = 0
lstm_start_time = None
frame_id = 0

cached_lstm_response = None
lstm_cache_until = 0.0

# =====================================================
# API ENDPOINT
# =====================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "message": "Flask API running"
    })


@app.route("/predict-lstm", methods=["POST"])
def predict_lstm():
    global sequence
    global hand_missing_count
    global lstm_start_time
    global frame_id
    global cached_lstm_response
    global lstm_cache_until

    payload = request.get_json(silent=True) or {}
    image = payload.get("image")

    if not image:
        return jsonify({
            "label": "No image",
            "confidence": 0.0,
            "mode": "LSTM",
            "status": "error"
        }), 400

    frame = decode_data_url(image)

    with lock:
        now = time.time()
        frame_id += 1

        if cached_lstm_response is not None and now < lstm_cache_until:
            return jsonify(cached_lstm_response)

        if lstm_start_time is not None and (now - lstm_start_time) > LSTM_TIMEOUT:
            sequence = []
            hand_missing_count = 0
            lstm_start_time = None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = hand_detector.detect_for_video(mp_image, frame_id)

        if not result.hand_landmarks:
            hand_missing_count += 1

            if hand_missing_count >= HAND_MISSING_LIMIT:
                sequence = []
                hand_missing_count = 0
                lstm_start_time = None

            return jsonify({
                "label": f"Collect {len(sequence)}/{seq_len}",
                "confidence": 0.0,
                "mode": "LSTM",
                "status": "collecting"
            })

        hand_missing_count = 0

        if lstm_start_time is None:
            lstm_start_time = now

        hand = result.hand_landmarks[0]

        coords = []

        for lm in hand:
            coords.extend([lm.x, lm.y, lm.z])

        coords = np.array(coords, dtype=np.float32)

        coords[0::3] = 1.0 - coords[0::3]

        sequence.append(coords)

        if len(sequence) > seq_len:
            sequence.pop(0)

        if len(sequence) < seq_len:
            return jsonify({
                "label": f"Collect {len(sequence)}/{seq_len}",
                "confidence": 0.0,
                "mode": "LSTM",
                "status": "collecting"
            })

        seq_arr = np.array(sequence, dtype=np.float32)

        seq_norm = normalize_keypoint_sequence(seq_arr)

        x = np.expand_dims(seq_norm, axis=0)

        x = apply_scaler(
            x,
            lstm_scaler_mean,
            lstm_scaler_scale
        )

        x = torch.tensor(
            x,
            dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            logits = lstm(x)
            probs = torch.softmax(logits, dim=1)

            pred_class = int(torch.argmax(probs, dim=1).item())
            conf = float(probs[0, pred_class].item())

        predicted = lstm_labels[pred_class]

        print(
            f"[LSTM] label={predicted}, conf={conf:.4f}"
        )

        sequence = []
        hand_missing_count = 0
        lstm_start_time = None

        if conf < LSTM_CONF_THRESHOLD or predicted == "none":
            return jsonify({
                "label": predicted,
                "confidence": conf,
                "mode": "LSTM",
                "status": "rejected"
            })

        response = {
            "label": predicted,
            "confidence": conf,
            "mode": "LSTM",
            "status": "accepted"
        }

        cached_lstm_response = dict(response)
        lstm_cache_until = now + RESULT_CACHE_SECONDS

        return jsonify(response)


@app.route("/predict-yolo", methods=["POST"])
def predict_yolo():
    payload = request.get_json(silent=True) or {}
    image = payload.get("image")

    if not image:
        return jsonify({
            "label": "No image",
            "confidence": 0.0,
            "mode": "YOLO",
            "status": "error"
        }), 400

    frame = decode_data_url(image)

    with lock:
        results = yolo(frame, verbose=False)[0]

        if len(results.boxes) == 0:
            return jsonify({
                "label": "Tidak terdeteksi",
                "confidence": 0.0,
                "mode": "YOLO",
                "status": "not_detected",
                "bbox": None
            })

        best_idx = int(torch.argmax(results.boxes.conf).item())

        conf = float(results.boxes.conf[best_idx])
        cls = int(results.boxes.cls[best_idx])

        label = yolo_labels[cls]

        x1, y1, x2, y2 = map(
            float,
            results.boxes.xyxy[best_idx]
        )

        return jsonify({
            "label": label,
            "confidence": conf,
            "mode": "YOLO",
            "status": "detected",
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": float(frame.shape[1]),
                "height": float(frame.shape[0]),
            }
        })


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("Loading models...")

    # Load YOLO
    yolo = YOLO(str(BASE_DIR / "models" / "best.pt"))
    yolo_labels = yolo.names

    # Load LSTM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(str(BASE_DIR / "models" / "lstm_model.pt"), map_location=device)

    state_dict = checkpoint["model_state_dict"]
    hidden1, hidden2 = infer_hidden_sizes(state_dict)
    bidirectional = infer_bidirectional(state_dict)

    lstm_labels = ["D", "I", "J", "Z", "none"]
    lstm = LSTMClassifier(input_size=63, hidden1=hidden1, hidden2=hidden2, num_classes=5, bidirectional=bidirectional).to(device)
    lstm.load_state_dict(state_dict)
    lstm.eval()

    lstm_scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32) if "scaler_mean" in checkpoint else None
    lstm_scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32) if "scaler_scale" in checkpoint else None
    seq_len = checkpoint.get("timesteps", 30)

    # Load MediaPipe
    base_options = python.BaseOptions(model_asset_path=str(BASE_DIR / "hand_landmarker.task"))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, running_mode=vision.RunningMode.VIDEO)
    hand_detector = vision.HandLandmarker.create_from_options(options)

    print("API READY at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
