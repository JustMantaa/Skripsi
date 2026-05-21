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


def infer_hidden_sizes(state_dict: dict) -> tuple[int, int]:
    h1 = state_dict["lstm1.weight_ih_l0"].shape[0] // 4
    h2 = state_dict["lstm2.weight_ih_l0"].shape[0] // 4
    return int(h1), int(h2)


def infer_bidirectional(state_dict: dict) -> bool:
    return "lstm2.weight_ih_l0_reverse" in state_dict


def normalize_keypoint_sequence(sequence: np.ndarray) -> np.ndarray:
    t, f = sequence.shape
    expected = 21 * 3
    if f != expected:
        raise ValueError(f"Expected {expected} features/frame, got {f}")

    seq3 = sequence.reshape(t, 21, 3).astype(np.float32)
    out = np.empty_like(seq3)
    wrist_ref = seq3[0, 0:1, :]

    for i in range(t):
        frame = seq3[i]
        centered = frame - wrist_ref

        scale = float(np.linalg.norm(frame[9] - frame[0]))
        if not np.isfinite(scale) or scale < 1e-6:
            distances = np.linalg.norm(centered[:, :2], axis=1)
            scale = float(np.max(distances)) if distances.size else 1.0
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0

        out[i] = centered / scale

    return out.reshape(t, f).astype(np.float32)


def apply_scaler(x: np.ndarray, mean_flat: np.ndarray | None, scale_flat: np.ndarray | None) -> np.ndarray:
    if mean_flat is None or scale_flat is None:
        # return shaped (1, t, f)
        return x.reshape(1, x.shape[0], x.shape[1]).astype(np.float32)

    x_flat = x.reshape(1, -1).astype(np.float32)
    safe_scale = np.where(scale_flat == 0, 1.0, scale_flat).astype(np.float32)
    x_scaled = (x_flat - mean_flat) / safe_scale
    return x_scaled.reshape(1, x.shape[0], x.shape[1]).astype(np.float32)


# =====================================================
# LOAD YOLO
# =====================================================
yolo = YOLO("output/best.pt")   # model huruf statis A-Y


# =====================================================
# LOAD LSTM
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("output/lstm_model.pt", map_location=device)

lstm_scaler_mean = None
lstm_scaler_scale = None

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
    features_per_frame = int(checkpoint.get("features_per_frame", 63))
    hidden1, hidden2 = infer_hidden_sizes(state_dict)
    bidirectional = bool(checkpoint.get("bidirectional", infer_bidirectional(state_dict)))

    lstm_labels = checkpoint.get("classes", ["J", "Z"])  # will verify below

    lstm = LSTMClassifier(
        input_size=features_per_frame,
        hidden1=hidden1,
        hidden2=hidden2,
        num_classes=len(lstm_labels),
        bidirectional=bidirectional,
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
    lstm = checkpoint.to(device)
    lstm_labels = ["J", "Z"]
    seq_len = 30
else:
    raise ValueError("Format output/lstm_model.pt tidak dikenali.")

# Enforce expected label set if checkpoint matches expected number of classes
EXPECTED_LABELS = ["D", "I", "J", "Z", "none"]
if len(lstm_labels) == len(EXPECTED_LABELS):
    lstm_labels = EXPECTED_LABELS

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
NO_DETECTION_LIMIT = 15    # jika YOLO tidak mendeteksi objek selama 15 frame (~0.5s @ 30fps) -> pindah LSTM
OUTPUT_DELAY_SECONDS = 3.5  # jeda perubahan teks output
LSTM_CONF_THRESHOLD = 0.70  # threshold confidence LSTM
LSTM_TIMEOUT = 3.0          # detik: timeout LSTM jika tangan tidak terdeteksi (mulai saat landmark detected)
HAND_MISSING_LIMIT = 5      # tolerance: reset sequence hanya jika tangan hilang >= 5 frame berturut-turut
OUTPUT_RESET_INTERVAL = 3.5 # detik: minimal jeda antara update teks output (debounce)
LSTM_FAIL_LIMIT = 2        # berapa kali LSTM boleh gagal sebelum revert ke YOLO
FORCE_LSTM_UNTIL_VALID = True  # jika True, jangan kembali ke YOLO sampai LSTM memberikan prediksi valid
FORCE_LSTM_MAX_TIMEOUT_RETRIES = 3  # berapa kali timeout LSTM diizinkan sebelum fallback ke YOLO
LSTM_FAIL_MAX_RETRIES = 5  # berapa kali kegagalan LSTM sebelum fallback ketika FORCE_LSTM_UNTIL_VALID=True


# =====================================================
# VARIABLE
# =====================================================
# mulai pada LSTM dulu (YOLO nonaktif sampai LSTM selesai)
mode = "LSTM"
no_detection_count = 0
sequence = []
result_text = "-"
display_lock_until = 0.0
yolo_enabled = False

# waktu mulai LSTM (None bila tidak dalam LSTM)
lstm_start_time = None

# counter untuk missing hand detection (tolerance terhadap frame yang miss)
hand_missing_count = 0
# last output tracking untuk debounce
last_output_time = 0.0
last_output_label = None
# counter kegagalan LSTM
lstm_fail_count = 0
# retry counters ketika FORCE_LSTM_UNTIL_VALID aktif
lstm_fail_retry_count = 0
timeout_retry_count = 0

print("[INFO] Starting in LSTM-first mode. YOLO disabled until first LSTM inference completes.")


# =====================================================
# CAMERA
# =====================================================
def open_camera():
    cap_candidate = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap_candidate.isOpened():
        cap_candidate = cv2.VideoCapture(0)
    if not cap_candidate.isOpened():
        return None
    cap_candidate.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[INFO] Camera opened at index 0")
    return cap_candidate


cap = open_camera()
if cap is None:
    raise RuntimeError("Kamera tidak bisa dibuka. Coba tutup aplikasi lain yang memakai kamera atau ubah index kamera.")

frame_id = 0


# =====================================================
# LOOP
# =====================================================
while True:

    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Gagal membaca frame dari kamera. Program dihentikan.")
        break

    now = time.time()

    # Frame original untuk LSTM (tanpa mirror)
    frame_original = frame.copy()
    # Frame mirrored untuk YOLO dan display (agar viewing natural)
    frame_display = cv2.flip(frame, 1)
    
    h, w, _ = frame.shape
    frame_id += 1

    # =================================================
    # MODE YOLO
    # =================================================

    if mode == "YOLO":

        results = None
        if yolo_enabled:
            results = yolo(frame_display, verbose=False)[0]

        # Jika ada deteksi, reset counter; jika tidak ada, tambahkan counter
        if results is not None and len(results.boxes) > 0:
            no_detection_count = 0

            # tampilkan deteksi tertinggi seperti sebelumnya
            conf = float(results.boxes.conf[0])
            cls = int(results.boxes.cls[0])
            label = yolo_labels[cls]
            x1, y1, x2, y2 = map(int, results.boxes.xyxy[0])
            cv2.rectangle(frame_display, (x1,y1), (x2,y2), (0,255,0), 2)

            if now >= display_lock_until:
                # debounce update: hanya update jika interval sudah lewat atau label berubah
                if (now - last_output_time) >= OUTPUT_RESET_INTERVAL or label != last_output_label:
                    result_text = f"{label} ({conf:.2f})"
                    last_output_time = now
                    last_output_label = label

            # track transitions (e.g., A -> J)
            try:
                if last_output_label is not None and last_output_label != '-' and last_output_label != result_text:
                    # extract label only from strings like 'A (0.98)' or 'Classification: A (0.98)'
                    cur_label = result_text.split()[0]
                    prev_label = last_output_label
                    if prev_label != cur_label:
                        # small filter: only act when both are single-letter classes
                        if len(prev_label) == 1 and len(cur_label) == 1:
                            pass
            except Exception:
                pass

        else:
            # tidak ada deteksi pada frame ini
            if yolo_enabled:
                no_detection_count += 1

        # switch ke LSTM jika tidak ada deteksi selama NO_DETECTION_LIMIT frame
        if no_detection_count >= NO_DETECTION_LIMIT:
            mode = "LSTM"
            sequence = []
            no_detection_count = 0
            lstm_start_time = None
            hand_missing_count = 0
            yolo_enabled = False
            print(f"[INFO] SWITCH -> LSTM (no_detection_count reached). frame_id={frame_id} time={now:.3f}")

    # =================================================
    # MODE LSTM
    # =================================================
    elif mode == "LSTM":

        rgb = cv2.cvtColor(frame_original, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # cek timeout LSTM
        if lstm_start_time is not None and (now - lstm_start_time) > LSTM_TIMEOUT:
            # Timeout saat mengumpulkan sequence; reset state internal dulu
            sequence = []
            hand_missing_count = 0
            lstm_start_time = None
            timeout_retry_count += 1

            if FORCE_LSTM_UNTIL_VALID:
                print(f"[INFO] LSTM timeout (retry {timeout_retry_count}/{FORCE_LSTM_MAX_TIMEOUT_RETRIES}) - staying in LSTM. frame_id={frame_id} time={now:.3f}")
                # jika sudah melewati retry limit, ijinkan fallback ke YOLO
                if timeout_retry_count >= FORCE_LSTM_MAX_TIMEOUT_RETRIES:
                    print(f"[WARN] FORCE_LSTM max timeout retries reached - fallback to YOLO. frame_id={frame_id}")
                    no_detection_count = 0
                    yolo_enabled = True
                    mode = "YOLO"
                    timeout_retry_count = 0
                # lanjutkan loop
                continue
            else:
                # fallback ke YOLO seperti sebelumnya
                no_detection_count = 0
                yolo_enabled = True
                mode = "YOLO"
                lstm_start_time = None
                continue

        result = hand_detector.detect_for_video(
            mp_image,
            frame_id
        )

        # Jika hand tidak terdeteksi: increment counter, reset hanya jika terlalu lama hilang
        if not result.hand_landmarks:
            hand_missing_count += 1

            # reset sequence hanya jika tangan hilang terlalu lama (>= HAND_MISSING_LIMIT frame)
            if hand_missing_count >= HAND_MISSING_LIMIT:
                sequence = []
                hand_missing_count = 0
        else:
            hand_missing_count = 0
            hand = result.hand_landmarks[0]

            # mulai timer saat landmark pertama terdeteksi (bukan saat masuk mode LSTM)
            if lstm_start_time is None:
                lstm_start_time = now

            # debug: log when we collect frames for sequence
            if (len(sequence) + 1) % 5 == 0 or len(sequence) < 3:
                print(f"[DEBUG] LSTM collecting frames: new_len={len(sequence)+1} frame_id={frame_id} time={now:.3f}")

            coords = []
            for lm in hand:
                coords.extend([lm.x, lm.y, lm.z])

            coords = np.array(coords, dtype=np.float32)

            # NORMALISASI (sesuaikan training kamu)
            wrist_x = coords[0]
            wrist_y = coords[1]

            for i in range(0, 63, 3):
                coords[i]   -= wrist_x
                coords[i+1] -= wrist_y

            sequence.append(coords)

            # sliding buffer: pastikan sequence tidak lebih dari seq_len frame
            if len(sequence) > seq_len:
                sequence.pop(0)

        # Jika 30 frame berturut-turut terkumpul
        if len(sequence) >= seq_len:

            seq_arr = np.array(sequence, dtype=np.float32)
            # normalisasi mengikuti realtime_test_lstm pipeline
            seq_norm = normalize_keypoint_sequence(seq_arr)
            x = apply_scaler(seq_norm, lstm_scaler_mean, lstm_scaler_scale)

            x = torch.tensor(x).to(device)

            with torch.no_grad():
                pred = lstm(x)
                probs = torch.softmax(pred, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                conf = float(probs[0, pred_class].item())

            predicted_label = lstm_labels[pred_class] if pred_class < len(lstm_labels) else "none"

            # Periksa threshold confidence LSTM
            print(f"[INFO] LSTM inference done. pred={predicted_label} conf={conf:.3f} frame_id={frame_id} time={now:.3f}")
            if conf < LSTM_CONF_THRESHOLD or predicted_label == "none":
                # LSTM gagal pada inferensi ini: coba ulang beberapa kali sebelum revert ke YOLO
                lstm_fail_count += 1
                sequence = []

                if lstm_fail_count >= LSTM_FAIL_LIMIT:
                    if FORCE_LSTM_UNTIL_VALID:
                        lstm_fail_retry_count += 1
                        print(f"[INFO] LSTM failed ({lstm_fail_retry_count}/{LSTM_FAIL_MAX_RETRIES}) but FORCE_LSTM_UNTIL_VALID=True, will retry. frame_id={frame_id} time={now:.3f}")
                        # jika melewati batas retry, ijinkan fallback
                        if lstm_fail_retry_count >= LSTM_FAIL_MAX_RETRIES:
                            print(f"[WARN] FORCE_LSTM max fail retries reached - fallback to YOLO. frame_id={frame_id}")
                            no_detection_count = 0
                            hand_missing_count = 0
                            yolo_enabled = True
                            mode = "YOLO"
                            lstm_start_time = None
                            lstm_fail_count = 0
                            lstm_fail_retry_count = 0
                        else:
                            # tetap di LSTM: reset counters untuk percobaan ulang
                            lstm_fail_count = 0
                            lstm_start_time = None
                    else:
                        # revert ke YOLO setelah kegagalan berulang
                        no_detection_count = 0
                        hand_missing_count = 0
                        yolo_enabled = True
                        mode = "YOLO"
                        lstm_start_time = None
                        print(f"[INFO] LSTM -> YOLO (none/low confidence, fail_count={lstm_fail_count}). frame_id={frame_id} time={now:.3f}")
                        lstm_fail_count = 0
                else:
                    # tetap di LSTM untuk mencoba lagi (jangan aktifkan YOLO)
                    print(f"[INFO] LSTM low/conf none — retrying ({lstm_fail_count}/{LSTM_FAIL_LIMIT}). frame_id={frame_id} time={now:.3f}")

            else:
                # hasil valid: reset fail counter dan keluarkan hasil
                lstm_fail_count = 0
                lstm_fail_retry_count = 0
                timeout_retry_count = 0
                voted = predicted_label

                # update teks berdasarkan voted result
                if now >= display_lock_until:
                    result_text = f"Classification: {voted} ({conf:.2f})"
                    display_lock_until = now + OUTPUT_DELAY_SECONDS
                    # update last output tracking
                    last_output_time = now
                    last_output_label = voted

                print(f"[INFO] LSTM -> YOLO (valid). voted={voted} conf={conf:.3f} frame_id={frame_id} time={now:.3f}")

                # Setelah inferensi selesai: reset sequence, reset counters, aktifkan YOLO
                sequence = []
                no_detection_count = 0
                hand_missing_count = 0
                yolo_enabled = True
                mode = "YOLO"
                lstm_start_time = None
                # (evidence saving removed)

    # =================================================
    # DISPLAY
    # =================================================
    cv2.putText(frame_display, f"Mode: {mode}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.putText(frame_display, result_text, (10,70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Hybrid YOLO + LSTM (LSTM-first)", frame_display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =====================================================
# END
# =====================================================
try:
    hand_detector.close()
except Exception:
    pass

cap.release()
cv2.destroyAllWindows()
