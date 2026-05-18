import base64
import threading
import time
from collections import deque
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
# SIBI CONFIG
# =====================================================
DYNAMIC_LETTERS = {"D", "I", "J", "Z"}
ALL_SIBI_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# =====================================================
# LSTM MODEL
# =====================================================
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, num_classes,
                 bidirectional=True, dropout_lstm=0.5, dropout_dense=0.3):
        super().__init__()
        self.bidirectional = bidirectional
        out_size = hidden2 * (2 if bidirectional else 1)
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_lstm)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2,
                             batch_first=True, bidirectional=bidirectional)
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
# HELPERS
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


# =====================================================
# LETTER-ONLY DETECTOR
# =====================================================
class LetterDetector:
    """
    Deteksi satu huruf SIBI per frame — tanpa akumulasi kata.
    
    States:
      IDLE       — tidak ada tangan
      STATIC     — YOLO aktif, huruf statis A-Z
      LSTM_RUN   — mengumpulkan sequence untuk D / J / Z
    """
    YOLO_CONF       = 0.72   # min confidence YOLO huruf statis
    YOLO_DYN_CONF   = 0.60   # min confidence YOLO untuk trigger LSTM
    LSTM_CONF       = 0.70   # min confidence LSTM diterima
    HAND_MISS_LIMIT = 8      # frame tanpa tangan → reset ke IDLE
    LSTM_TIMEOUT    = 12.0   # detik max tunggu sequence LSTM

    def __init__(self, yolo, lstm, hand_detector,
                 lstm_labels, yolo_labels, seq_len,
                 scaler_mean, scaler_scale, device):
        self.yolo = yolo
        self.lstm = lstm
        self.hand_detector = hand_detector
        self.lstm_labels = lstm_labels
        self.yolo_labels = yolo_labels
        self.seq_len = seq_len
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale
        self.device = device

        self.state = "IDLE"
        self.sequence = []
        self.hand_miss = 0
        self.lstm_start = None
        self.frame_id = 0

        # Smooth display: hold last accepted result for N seconds
        self.hold_letter = "-"
        self.hold_conf   = 0.0
        self.hold_source = "-"       # "YOLO" | "LSTM"
        self.hold_until  = 0.0
        self.hold_bbox   = None
        self.HOLD_SECS   = 1.2       # detik tahan hasil terakhir

    def _reset_lstm(self):
        self.sequence = []
        self.lstm_start = None
        self.state = "STATIC"

    def process(self, frame: np.ndarray) -> dict:
        self.frame_id += 1
        now = time.time()
        h, w = frame.shape[:2]

        # ── 1. MediaPipe hand detection ──────────────────────────
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_res = self.hand_detector.detect_for_video(mp_img, self.frame_id)
        hand_ok = bool(mp_res.hand_landmarks)

        if not hand_ok:
            self.hand_miss += 1
            if self.hand_miss >= self.HAND_MISS_LIMIT:
                self.state = "IDLE"
                self.sequence = []
                self.lstm_start = None
            # Return held result or empty
            return self._response(now, w, h, hand_ok=False)

        self.hand_miss = 0
        if self.state == "IDLE":
            self.state = "STATIC"

        # ── 2. YOLO ─────────────────────────────────────────────
        yolo_res = self.yolo(frame, verbose=False)[0]
        yolo_letter, yolo_conf, yolo_bbox = None, 0.0, None

        if len(yolo_res.boxes) > 0:
            best = int(torch.argmax(yolo_res.boxes.conf).item()) \
                   if len(yolo_res.boxes) > 1 else 0
            yolo_conf = float(yolo_res.boxes.conf[best])
            cls = int(yolo_res.boxes.cls[best])
            lbl = self.yolo_labels[cls].upper()
            if lbl in ALL_SIBI_LETTERS:
                yolo_letter = lbl
                x1, y1, x2, y2 = map(float, yolo_res.boxes.xyxy[best])
                yolo_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "width": float(w), "height": float(h)}

        # ── 3. State logic ───────────────────────────────────────
        if self.state == "STATIC":
            if yolo_letter and yolo_conf >= self.YOLO_CONF:
                if yolo_letter in DYNAMIC_LETTERS and yolo_conf >= self.YOLO_DYN_CONF:
                    # Trigger LSTM
                    self.state = "LSTM_RUN"
                    self.sequence = []
                    self.lstm_start = now
                else:
                    # Accept static letter immediately
                    self._hold(yolo_letter, yolo_conf, "YOLO", yolo_bbox, now)

        if self.state == "LSTM_RUN":
            # Timeout guard
            if self.lstm_start and (now - self.lstm_start) > self.LSTM_TIMEOUT:
                self._reset_lstm()
                return self._response(now, w, h, hand_ok=True)

            # Collect keypoints
            hand = mp_res.hand_landmarks[0]
            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])
            coords = np.array(coords, dtype=np.float32)
            coords[0::3] = 1.0 - coords[0::3]
            self.sequence.append(coords)
            if len(self.sequence) > self.seq_len:
                self.sequence.pop(0)

            if len(self.sequence) >= self.seq_len:
                pred, conf = self._run_lstm()
                if conf >= self.LSTM_CONF and pred in DYNAMIC_LETTERS:
                    self._hold(pred, conf, "LSTM", None, now)
                self._reset_lstm()

        return self._response(now, w, h, hand_ok=True, yolo_letter=yolo_letter,
                              yolo_conf=yolo_conf, yolo_bbox=yolo_bbox)

    def _run_lstm(self):
        seq_arr  = np.array(self.sequence, dtype=np.float32)
        seq_norm = normalize_keypoint_sequence(seq_arr)
        x = apply_scaler(np.expand_dims(seq_norm, 0),
                         self.scaler_mean, self.scaler_scale)
        x_t = torch.tensor(x).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.lstm(x_t), dim=1)
            cls   = int(torch.argmax(probs, dim=1).item())
            conf  = float(probs[0, cls].item())
        label = self.lstm_labels[cls].upper()
        print(f"[LSTM] {label} {conf:.3f}")
        return label, conf

    def _hold(self, letter, conf, source, bbox, now):
        self.hold_letter = letter
        self.hold_conf   = conf
        self.hold_source = source
        self.hold_bbox   = bbox
        self.hold_until  = now + self.HOLD_SECS

    def _response(self, now, w, h, hand_ok,
                  yolo_letter=None, yolo_conf=0.0, yolo_bbox=None):
        # During LSTM collection, keep hold result visible
        active_hold = (now < self.hold_until)

        letter  = self.hold_letter if active_hold else ("-" if not hand_ok else "?")
        conf    = self.hold_conf   if active_hold else 0.0
        source  = self.hold_source if active_hold else ("-" if not hand_ok else "YOLO")
        bbox    = self.hold_bbox   if active_hold else yolo_bbox

        return {
            "state":         self.state,
            "letter":        letter,
            "confidence":    round(conf, 4),
            "source":        source,
            "bbox":          bbox,
            "hand_detected": hand_ok,
            "lstm_progress": len(self.sequence),
            "lstm_total":    self.seq_len,
        }


# =====================================================
# FLASK
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
CORS(app)
lock = threading.Lock()
detector: LetterDetector = None


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "No image"}), 400
    frame = decode_data_url(image_data)
    with lock:
        result = detector.process(frame)
    return jsonify(result)


@app.route("/config", methods=["GET"])
def get_config():
    return jsonify({
        "yolo_conf":      LetterDetector.YOLO_CONF,
        "yolo_dyn_conf":  LetterDetector.YOLO_DYN_CONF,
        "lstm_conf":      LetterDetector.LSTM_CONF,
        "hold_secs":      detector.HOLD_SECS,
        "seq_len":        detector.seq_len,
        "dynamic_letters": list(DYNAMIC_LETTERS),
    })


@app.route("/config", methods=["POST"])
def set_config():
    p = request.get_json(silent=True) or {}
    with lock:
        if "yolo_conf"     in p: LetterDetector.YOLO_CONF     = float(p["yolo_conf"])
        if "yolo_dyn_conf" in p: LetterDetector.YOLO_DYN_CONF = float(p["yolo_dyn_conf"])
        if "lstm_conf"     in p: LetterDetector.LSTM_CONF      = float(p["lstm_conf"])
        if "hold_secs"     in p: detector.HOLD_SECS            = float(p["hold_secs"])
    return jsonify({"ok": True})


@app.route("/reset", methods=["POST"])
def reset():
    with lock:
        detector.state = "IDLE"
        detector.sequence = []
        detector.hand_miss = 0
        detector.lstm_start = None
        detector.hold_letter = "-"
        detector.hold_conf = 0.0
        detector.hold_until = 0.0
    return jsonify({"ok": True})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "state": detector.state})


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(str(BASE_DIR / "models" / "best.pt"))
    yolo_labels = yolo.names

    ckpt = torch.load(str(BASE_DIR / "models" / "lstm_model.pt"), map_location=device)
    sd = ckpt["model_state_dict"]
    h1, h2 = infer_hidden_sizes(sd)
    bi = infer_bidirectional(sd)
    lstm_labels = ckpt.get("labels", ["D", "I", "J", "Z", "none"])
    seq_len = ckpt.get("timesteps", 30)
    s_mean  = np.asarray(ckpt["scaler_mean"],  dtype=np.float32) if "scaler_mean"  in ckpt else None
    s_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float32) if "scaler_scale" in ckpt else None

    lstm = LSTMClassifier(63, h1, h2, len(lstm_labels), bi).to(device)
    lstm.load_state_dict(sd)
    lstm.eval()

    base_opt = python.BaseOptions(model_asset_path=str(BASE_DIR / "hand_landmarker.task"))
    opts = vision.HandLandmarkerOptions(
        base_options=base_opt, num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO,
    )
    hand_det = vision.HandLandmarker.create_from_options(opts)

    detector = LetterDetector(yolo, lstm, hand_det, lstm_labels,
                              yolo_labels, seq_len, s_mean, s_scale, device)

    print(f"Ready  →  http://127.0.0.1:5000   (device: {device})")
    app.run(host="127.0.0.1", port=5000, debug=False)