import argparse
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

# ============================================================================
# KONFIGURASI
# ============================================================================
YOLO_MODEL_PATH = "output/best.pt"
CAMERA_INDEX = 0

def main():
    parser = argparse.ArgumentParser(description="Test YOLO detection dari webcam atau video")
    parser.add_argument("--model", default=YOLO_MODEL_PATH, help="Path model YOLO .pt")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Index kamera (default 0)")
    parser.add_argument("--input", help="Path video file (jika kosong gunakan webcam)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    parser.add_argument("--imgsz", type=int, default=960, help="Input size YOLO (default 960)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (default 0.45)")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval update hasil dalam detik (default 2.0)")
    parser.add_argument("--smooth-window", type=int, default=5, help="Jumlah prediksi terakhir untuk smoothing (default 5)")
    args = parser.parse_args()

    # Validasi file
    yolo_path = Path(args.model)
    if not yolo_path.exists():
        raise FileNotFoundError(f"Model YOLO tidak ditemukan: {yolo_path}")

    # Load model
    print(f"Loading YOLO model from: {yolo_path}")
    yolo_model = YOLO(str(yolo_path))
    print("✓ Model loaded successfully!")
    print(f"  Classes: {yolo_model.names}")
    print()

    # Buka video/webcam
    use_webcam = not bool(args.input)
    if args.input:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise ValueError(f"Tidak bisa membuka video: {args.input}")
        print(f"Opening video: {args.input}")
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise ValueError(f"Tidak bisa membuka webcam index {args.camera}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"Opening webcam: {args.camera}")

    print("Press 'q' to quit")
    print()

    last_pred_time = 0.0
    last_label = "No Detection"
    last_conf = 0.0
    last_det_count = 0
    pred_history = deque(maxlen=max(1, args.smooth_window))
    cached_boxes = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror hanya untuk mode webcam agar gerakan terasa natural.
            if use_webcam:
                            frame = cv2.flip(frame, 1)

            now = time.monotonic()

            # Jalankan deteksi hanya setiap interval agar hasil update per 2 detik.
            if (now - last_pred_time) >= args.interval:
                results = yolo_model(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
                if results and len(results[0].boxes) > 0:
                    # Simpan box terbaru agar bbox dan teks selalu render dari sumber data yang sama.
                    cached_boxes = [b for b in results[0].boxes]
                    best_box = max(results[0].boxes, key=lambda b: float(b.conf[0].item()))
                    class_id = int(best_box.cls[0].item())
                    curr_label = str(yolo_model.names[class_id])
                    curr_conf = float(best_box.conf[0].item())
                    pred_history.append((curr_label, curr_conf))

                    # Smoothing sederhana: pilih label paling sering, tie-break by confidence rata-rata.
                    label_scores = {}
                    label_counts = {}
                    for lbl, cf in pred_history:
                        label_scores[lbl] = label_scores.get(lbl, 0.0) + cf
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1

                    best_label = max(
                        label_counts,
                        key=lambda lbl: (label_counts[lbl], label_scores[lbl] / label_counts[lbl]),
                    )
                    avg_conf = label_scores[best_label] / max(1, label_counts[best_label])

                    last_label = best_label
                    last_conf = float(avg_conf)
                    last_det_count = len(results[0].boxes)
                else:
                    cached_boxes = []
                    pred_history.append(("No Detection", 0.0))
                    last_label = "No Detection"
                    last_conf = 0.0
                    last_det_count = 0
                last_pred_time = now

            # Tampilkan bounding box dari cache hasil deteksi terbaru.
            if cached_boxes:
                for box in cached_boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0].item())
                    class_id = int(box.cls[0].item())
                    class_name = str(yolo_model.names[class_id])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{class_name} {conf:.2f}",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            cv2.putText(
                frame,
                f"Hasil: {last_label} ({last_conf:.2f})",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("YOLO Detection Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    main()
