import cv2
import torch
import torch.nn as nn
import numpy as np
import time
from ultralytics import YOLO
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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

# =====================================================
# LOAD YOLO
# =====================================================
yolo = YOLO("output/best.pt")   # model huruf statis A-Y

# =====================================================
# LOAD LSTM
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"  # paksa pakai CPU saja untuk kompatibilitas lebih luas

checkpoint = torch.load("output/lstm_model.pt", map_location=device)

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

    # Backward compatibility jika checkpoint lama tidak punya fc_mid.
    if "fc_mid.weight" not in state_dict:
        lstm.fc_mid = nn.Identity()
        lstm.fc2 = nn.Linear(32, len(lstm_labels)).to(device)

    lstm.load_state_dict(state_dict)

    if "scaler_mean" in checkpoint and "scaler_scale" in checkpoint:
        lstm_scaler_mean = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
        lstm_scaler_scale = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)

    seq_len = int(checkpoint.get("timesteps", 30))
elif isinstance(checkpoint, nn.Module):
    # Support model yang disimpan langsung sebagai objek nn.Module.
    lstm = checkpoint.to(device)
    lstm_labels = ["J", "Z"]
    seq_len = 30
else:
    raise ValueError("Format output/lstm_model.pt tidak dikenali.")

lstm.eval()

# =====================================================
# LABEL
# =====================================================
yolo_labels = yolo.names

# =====================================================
# MEDIAPIPE TASK API
# =====================================================
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

hand_detector = vision.HandLandmarker.create_from_options(options)

# =====================================================
# PARAMETER
# =====================================================
threshold = 0.5         # confidence YOLO
low_conf_limit = 5       # berapa frame rendah berturut
OUTPUT_DELAY_SECONDS = 1.5  # jeda perubahan teks output

# =====================================================
# VARIABLE
# =====================================================
mode = "YOLO"
low_conf_count = 0
sequence = []
display_text = "-"
display_lock_until = 0.0

# =====================================================
# CAMERA
# =====================================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_id = 0

# =====================================================
# LOOP
# =====================================================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_id += 1

    # =================================================
    # MODE YOLO
    # =================================================
    if mode == "YOLO":

        results = yolo(frame, verbose=False)[0]

        if len(results.boxes) > 0:

            # ambil prediksi tertinggi
            conf = float(results.boxes.conf[0])
            cls = int(results.boxes.cls[0])

            label = yolo_labels[cls]

            x1, y1, x2, y2 = map(int, results.boxes.xyxy[0])

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            if now >= display_lock_until:
                display_text = f"{label} ({conf:.2f})"

            # confidence rendah
            if conf < threshold:
                low_conf_count += 1
            else:
                low_conf_count = 0

            # switch ke LSTM
            if low_conf_count >= low_conf_limit:
                mode = "LSTM"
                sequence = []
                display_text = "Switch to LSTM"
                display_lock_until = now + OUTPUT_DELAY_SECONDS

        else:
            low_conf_count += 1

    # =================================================
    # MODE LSTM
    # =================================================
    elif mode == "LSTM":

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = hand_detector.detect_for_video(
            mp_image,
            frame_id
        )

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            coords = []

            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])

            # total 63 fitur
            coords = np.array(coords, dtype=np.float32)

            # =========================================
            # NORMALISASI (sesuaikan training kamu)
            # =========================================
            wrist_x = coords[0]
            wrist_y = coords[1]

            for i in range(0, 63, 3):
                coords[i]   -= wrist_x
                coords[i+1] -= wrist_y

            sequence.append(coords)

            if now >= display_lock_until:
                display_text = f"LSTM Collect {len(sequence)}/{seq_len}"

        # =============================================
        # Jika 30 frame sudah terkumpul
        # =============================================
        if len(sequence) == seq_len:

            x = np.array(sequence, dtype=np.float32)
            x = np.expand_dims(x, axis=0)
            x = apply_scaler(x, lstm_scaler_mean, lstm_scaler_scale)

            x = torch.tensor(x).to(device)

            with torch.no_grad():
                pred = lstm(x)
                pred_class = torch.argmax(pred, dim=1).item()

            display_text = f"LSTM: {lstm_labels[pred_class]}"
            display_lock_until = now + OUTPUT_DELAY_SECONDS

            # reset kembali ke YOLO
            mode = "YOLO"
            low_conf_count = 0
            sequence = []

    # =================================================
    # DISPLAY
    # =================================================
    cv2.putText(frame, f"Mode: {mode}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.putText(frame, display_text, (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Hybrid YOLO + LSTM", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =====================================================
# END
# =====================================================
cap.release()
cv2.destroyAllWindows()