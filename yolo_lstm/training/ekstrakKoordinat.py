import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd
from tqdm import tqdm

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==============================
# CONFIG
# ==============================
DATA_PATH = "dataset"
LABELS = ["J", "Z", "none"]  # Sesuaikan dengan folder di dataset
SEQUENCE_LENGTH = 30
MODEL_PATH = "hand_landmarker.task"
OUTPUT_CSV = "dataset_lstm.csv"

# ==============================
# LOAD MODEL
# ==============================
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# ==============================
# EXTRACT KEYPOINT
# ==============================
def extract_keypoints(frame):
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        keypoints = []

        for lm in hand:
            keypoints.extend([lm.x, lm.y, lm.z])

        return np.array(keypoints)

    return np.zeros(63)

# ==============================
# VIDEO → SEQUENCE
# ==============================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = extract_keypoints(frame)
        frames.append(keypoints)

    cap.release()

    frames = np.array(frames)

    # Samakan panjang
    if len(frames) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(frames), 63))
        frames = np.vstack((frames, pad))
    else:
        idx = np.linspace(0, len(frames)-1, SEQUENCE_LENGTH).astype(int)
        frames = frames[idx]

    return frames

# ==============================
# MAIN LOOP
# ==============================
X = []
y = []

for label in LABELS:
    folder_aug = os.path.join(DATA_PATH, f"{label}_aug")
    folder_raw = os.path.join(DATA_PATH, label)
    folder = folder_aug if os.path.isdir(folder_aug) else folder_raw

    print(f"Processing {label}...")

    for file in tqdm(os.listdir(folder)):
        if file.endswith(".mp4"):
            path = os.path.join(folder, file)

            sequence = process_video(path)

            X.append(sequence)
            y.append(label)

X = np.array(X)
y = np.array(y)

print("Shape:", X.shape)

# ==============================
# FLATTEN + CSV
# ==============================
X_flat = X.reshape(X.shape[0], -1)

df = pd.DataFrame(X_flat)
df.insert(0, "label", y)

df.to_csv(OUTPUT_CSV, index=False)

print("✅ Dataset berhasil dibuat!")