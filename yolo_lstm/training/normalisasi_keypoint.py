import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
FEATURES_PER_FRAME = NUM_LANDMARKS * COORDS_PER_LANDMARK
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
BASE_DIR = Path(__file__).resolve().parents[1]


def _reshape_sequence(row_values: np.ndarray, timesteps: int) -> np.ndarray:
    expected = timesteps * FEATURES_PER_FRAME
    if row_values.size != expected:
        raise ValueError(
            f"Expected {expected} feature values for timesteps={timesteps}, got {row_values.size}."
        )
    return row_values.reshape(timesteps, NUM_LANDMARKS, COORDS_PER_LANDMARK).astype(np.float32)


def _normalize_frame(frame: np.ndarray, wrist_ref: np.ndarray) -> np.ndarray:
    """
    Normalize one frame of hand keypoints.

    Steps:
    1. Center coordinates using a sequence-level wrist reference.
    2. Scale coordinates using wrist-to-middle_mcp distance.
    3. Fallback to max landmark distance if reference scale is too small.
    """
    centered = frame.copy()
    centered = centered - wrist_ref

    scale = np.linalg.norm(frame[MIDDLE_MCP_INDEX] - frame[WRIST_INDEX])
    if not np.isfinite(scale) or scale < 1e-6:
        distances = np.linalg.norm(centered[:, :2], axis=1)
        scale = float(np.max(distances)) if distances.size else 1.0

    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0

    centered = centered / scale
    return centered


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    sequence shape: (timesteps, 63)
    return shape: (timesteps, 63)

    Note:
    - Centering menggunakan wrist frame pertama untuk menjaga informasi
      perpindahan tangan antar frame (penting untuk gesture dinamis seperti J/Z).
    """
    timesteps = sequence.shape[0]
    reshaped = sequence.reshape(timesteps, NUM_LANDMARKS, COORDS_PER_LANDMARK)
    wrist_ref = reshaped[0, WRIST_INDEX:WRIST_INDEX + 1, :]

    normalized_frames = []
    for frame in reshaped:
        normalized_frames.append(_normalize_frame(frame, wrist_ref))

    normalized = np.stack(normalized_frames, axis=0)
    return normalized.reshape(timesteps, FEATURES_PER_FRAME).astype(np.float32)


def normalize_dataframe(df: pd.DataFrame, timesteps: int = 30) -> pd.DataFrame:
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column.")

    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) != timesteps * FEATURES_PER_FRAME:
        raise ValueError(
            f"Expected {timesteps * FEATURES_PER_FRAME} feature columns, got {len(feature_cols)}."
        )

    normalized_rows = []
    for _, row in df.iterrows():
        label = row["label"]
        values = row[feature_cols].to_numpy(dtype=np.float32)
        seq = _reshape_sequence(values, timesteps)
        norm_seq = normalize_sequence(seq.reshape(timesteps, FEATURES_PER_FRAME))
        normalized_rows.append([label, *norm_seq.reshape(-1).tolist()])

    columns = ["label", *feature_cols]
    return pd.DataFrame(normalized_rows, columns=columns)


def normalize_csv(input_csv: str, output_csv: str, timesteps: int = 30) -> Path:
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    normalized_df = normalize_dataframe(df, timesteps=timesteps)
    normalized_df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize hand keypoint CSV using centering and scaling.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(BASE_DIR / "output" / "dataset.csv"),
        help="Path to input CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(BASE_DIR / "output" / "dataset_normalized.csv"),
        help="Path to save normalized CSV.",
    )
    parser.add_argument("--timesteps", type=int, default=30, help="Number of timesteps per sample.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    if not input_csv.is_absolute():
        input_csv = BASE_DIR / input_csv
    if not output_csv.is_absolute():
        output_csv = BASE_DIR / output_csv

    out = normalize_csv(str(input_csv), str(output_csv), timesteps=args.timesteps)
    print(f"Normalized dataset saved to: {out}")


if __name__ == "__main__":
    main()
