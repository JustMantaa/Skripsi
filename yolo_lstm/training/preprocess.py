import argparse
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
BASE_DIR = Path(__file__).resolve().parents[1]


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


def normalize_video(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    target_fps: float,
    mirror: bool,
):
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

    # Ambil ukuran asli frame
    frame_height, frame_width = frames[0].shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Tidak bisa membuat output video: {output_path}")

    for i in indices:
        frame = frames[int(i)]
        if mirror:
            frame = cv2.flip(frame, 1)
        # Tidak perlu resize, gunakan ukuran asli
        writer.write(frame)

    writer.release()


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    classes: list[str],
    width: int,
    height: int,
    fps: float,
    mirror: bool,
):
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
                normalize_video(video_path, out_path, width, height, fps, mirror)
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
    parser = argparse.ArgumentParser(description="Preprocess video ke FPS tertentu tanpa mengubah ukuran frame")
    parser.add_argument("--dataset-dir", default=str(BASE_DIR / "dataset"), help="Folder dataset input")
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "output" / "preprocessed_fps_only"),
        help="Folder output video normalisasi",
    )
    parser.add_argument(
        "--classes",
        default="D,I,J,R,M,N,none,U,V,W,Z",
        help="Daftar kelas dipisah koma, contoh: D,I,J,M,N,none,U,V,W,Z",
    )
    # width dan height tidak diperlukan lagi, tapi tetap diterima agar kompatibel
    parser.add_argument("--width", type=int, default=0, help="(Diabaikan) Lebar output")
    parser.add_argument("--height", type=int, default=0, help="(Diabaikan) Tinggi output")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS output")
    parser.add_argument("--mirror", dest="mirror", action="store_true", help="Mirror horizontal frame output")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", help="Nonaktifkan mirror horizontal")
    parser.set_defaults(mirror=False)
    return parser.parse_args()


def main():
    args = parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not dataset_dir.is_absolute():
        dataset_dir = BASE_DIR / dataset_dir
    if not output_dir.is_absolute():
        output_dir = BASE_DIR / output_dir

    process_dataset(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        classes=classes,
        width=args.width,  # tetap diteruskan agar signature tidak berubah
        height=args.height,
        fps=args.fps,
        mirror=args.mirror,
    )


if __name__ == "__main__":
    main()
