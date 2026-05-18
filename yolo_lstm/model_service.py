import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

try:
    # when package-imported (e.g., `import yolo_lstm.model_service`)
    from .utlis import decode_data_url, normalize_keypoint_sequence, apply_scaler, build_response
except Exception:
    # when run as a script from the same folder (e.g., `python inference_api_v2.py`)
    from utlis import decode_data_url, normalize_keypoint_sequence, apply_scaler, build_response


# ------------------
# LSTM CLASSIFIER
# ------------------
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


def infer_hidden_sizes(state_dict):
    h1 = state_dict["lstm1.weight_ih_l0"].shape[0] // 4
    h2 = state_dict["lstm2.weight_ih_l0"].shape[0] // 4
    return h1, h2


def infer_bidirectional(state_dict):
    return "lstm2.weight_ih_l0_reverse" in state_dict


# ------------------
# CONFIG
# ------------------
BASE_DIR = Path(__file__).resolve().parent
NO_DETECTION_LIMIT = 3
RESULT_CACHE_SECONDS = 3.0
LSTM_CONF_THRESHOLD = 0.70
LSTM_TIMEOUT = 15.0
HAND_MISSING_LIMIT = 5


# ------------------
# STATE
# ------------------
lock = threading.Lock()
mode = "LSTM"
yolo_enabled = False
no_detection_count = 0
sequence = []
hand_missing_count = 0
lstm_start_time = None
frame_id = 0
cached_response = None
cache_until = 0.0


# ------------------
# LOAD MODELS
# ------------------
print("Loading models (model_service)...")

# YOLO
yolo = YOLO(str(BASE_DIR / "models" / "best.pt"))
yolo_labels = yolo.names

# LSTM
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

# MediaPipe
base_options = python.BaseOptions(model_asset_path=str(BASE_DIR / "hand_landmarker.task"))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, running_mode=vision.RunningMode.VIDEO)
hand_detector = vision.HandLandmarker.create_from_options(options)

print("Models loaded in model_service")


def predict_from_image_dataurl(data_url):
    """Main prediction entrypoint. Accepts a data URL string and returns a dict response."""
    global mode, yolo_enabled, no_detection_count, sequence, hand_missing_count
    global lstm_start_time, frame_id, cached_response, cache_until

    frame = decode_data_url(data_url)

    with lock:
        now = time.time()
        frame_id += 1

        if cached_response is not None and now < cache_until:
            cached = dict(cached_response)
            cached["mode"] = mode
            return cached

        response = build_response(mode_name=mode)

        # ===== MODE: YOLO =====
        if mode == "YOLO":
            results = yolo(frame, verbose=False)[0]

            if len(results.boxes) > 0:
                no_detection_count = 0
                conf = float(results.boxes.conf[0])
                cls = int(results.boxes.cls[0])
                label = yolo_labels[cls]
                x1, y1, x2, y2 = map(float, results.boxes.xyxy[0])

                response = build_response(
                    label=label,
                    confidence=conf,
                    mode_name="YOLO",
                    bbox={
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "width": float(frame.shape[1]),
                        "height": float(frame.shape[0]),
                    },
                )
                cached_response = dict(response)
                cache_until = now + RESULT_CACHE_SECONDS
            else:
                no_detection_count += 1
                response["label"] = "Tidak terdeteksi"
                response["mode"] = "YOLO"

                if no_detection_count >= NO_DETECTION_LIMIT:
                    mode = "LSTM"
                    yolo_enabled = False
                    sequence = []
                    hand_missing_count = 0
                    lstm_start_time = None
                    no_detection_count = 0
                    cached_response = None
                    cache_until = 0.0

        # ===== MODE: LSTM =====
        elif mode == "LSTM":
            if lstm_start_time is not None and (now - lstm_start_time) > LSTM_TIMEOUT:
                sequence = []
                hand_missing_count = 0
                lstm_start_time = None

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = hand_detector.detect_for_video(mp_image, frame_id)

            if not result.hand_landmarks:
                hand_missing_count += 1
                response["label"] = f"Collect {len(sequence)}/{seq_len}"
                response["mode"] = "LSTM"

                if hand_missing_count >= HAND_MISSING_LIMIT:
                    sequence = []
                    hand_missing_count = 0

            else:
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

                response["label"] = f"Collect {len(sequence)}/{seq_len}"
                response["mode"] = "LSTM"

                if len(sequence) >= seq_len:
                    seq_arr = np.array(sequence, dtype=np.float32)
                    seq_norm = normalize_keypoint_sequence(seq_arr)
                    x = np.expand_dims(seq_norm, axis=0)
                    x = apply_scaler(x, lstm_scaler_mean, lstm_scaler_scale)
                    x = torch.tensor(x).to(device)

                    with torch.no_grad():
                        logits = lstm(x)
                        probs = torch.softmax(logits, dim=1)
                        pred_class = int(torch.argmax(probs, dim=1).item())
                        conf = float(probs[0, pred_class].item())

                    predicted = lstm_labels[pred_class]
                    print(
                        f"[DEBUG] LSTM Output: label={predicted}, conf={conf:.4f}, "
                        f"threshold={LSTM_CONF_THRESHOLD}, all_probs={probs[0].cpu().numpy()}"
                    )

                    if conf < LSTM_CONF_THRESHOLD or predicted == "none":
                        print(
                            f"[DEBUG] LSTM rejected: conf={conf:.4f}, label={predicted}. "
                            f"Switching to YOLO."
                        )
                        mode = "YOLO"
                        yolo_enabled = True
                        sequence = []
                        hand_missing_count = 0
                        lstm_start_time = None
                        no_detection_count = 0
                        cached_response = None
                        cache_until = 0.0
                        response = build_response(label="none", confidence=conf, mode_name="YOLO")
                    else:
                        print(f"[DEBUG] LSTM accepted: {predicted} with conf={conf:.4f}")
                        response = build_response(label=predicted, confidence=conf, mode_name="LSTM")
                        cached_response = dict(response)
                        cache_until = now + RESULT_CACHE_SECONDS
                        sequence = []
                        hand_missing_count = 0
                        lstm_start_time = None

        return response
