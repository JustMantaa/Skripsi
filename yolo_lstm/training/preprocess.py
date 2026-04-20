import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def list_videos(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])


def resample_indices(num_frames: int, src_fps: float, target_fps: float):
    if num_frames <= 0:
        return np.array([], dtype=np.int32)

    if src_fps is None or not np.isfinite(src_fps) or src_fps <= 1e-3:
        src_fps = target_fps

    # Keep source duration, change fps by selecting frames uniformly.
    target_count = max(1, int(round(num_frames * (target_fps / src_fps))))
    idx = np.linspace(0, num_frames - 1, target_count)
    return np.clip(np.round(idx).astype(np.int32), 0, num_frames - 1)


def normalize_video(input_path: Path, output_path: Path, width: int, height: int, target_fps: float):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {input_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS))

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Video kosong / gagal dibaca: {input_path}")

    indices = resample_indices(len(frames), src_fps, target_fps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Tidak bisa membuat output video: {output_path}")

    for i in indices:
        frame = frames[int(i)]
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        writer.write(frame)

    writer.release()


def process_dataset(dataset_dir: Path, output_dir: Path, classes: list[str], width: int, height: int, fps: float):
    total = 0
    success = 0

    for cls in classes:
        in_dir = dataset_dir / cls
        out_dir = output_dir / cls
        videos = list_videos(in_dir)
        print(f"[{cls}] ditemukan {len(videos)} video")

        for idx, video_path in enumerate(videos, start=1):
            total += 1
            out_path = out_dir / f"{video_path.stem}.mp4"
            try:
                normalize_video(video_path, out_path, width, height, fps)
                success += 1
                if idx % 10 == 0 or idx == len(videos):
                    print(f"  {idx}/{len(videos)} selesai")
            except Exception as e:
                print(f"  Gagal: {video_path.name} -> {e}")

    print("\nRingkasan:")
    print(f"Total video: {total}")
    print(f"Berhasil: {success}")
    print(f"Gagal: {total - success}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess video ke 640x640 dan 30 FPS")
    parser.add_argument("--dataset-dir", default="dataset", help="Folder dataset input")
    parser.add_argument(
        "--output-dir",
        default="output/preprocessed_640x640_30fps",
        help="Folder output video normalisasi",
    )
    parser.add_argument("--classes", default="J,Z,none", help="Daftar kelas dipisah koma, contoh: J,Z,none")
    parser.add_argument("--width", type=int, default=640, help="Lebar output")
    parser.add_argument("--height", type=int, default=640, help="Tinggi output")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS output")
    return parser.parse_args()


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    process_dataset(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        classes=classes,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
