from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture anti-spoof face crops from camera."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/anti_spoof"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--label",
        type=str,
        choices=["live", "spoof"],
        required=True,
        help="Class label to capture.",
    )
    parser.add_argument(
        "--person",
        type=str,
        default="user",
        help="Person/session name for file prefix.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=150,
        help="How many images to save.",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=0.20,
        help="Minimum time between captures.",
    )
    return parser


def detect_largest_face(gray, detector):
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    return int(x), int(y), int(w), int(h)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = args.output_dir / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

    print("Capture started.")
    print("Controls: [space]=save frame, [a]=auto mode on/off, [q]=quit")
    print(f"Label: {args.label}, target: {args.target_count}")

    saved = 0
    auto_mode = True
    last_save_at = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = detect_largest_face(gray, detector)

            crop = None
            if face is not None:
                x, y, w, h = face
                pad_x = int(w * 0.20)
                pad_y = int(h * 0.20)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(frame.shape[1], x + w + pad_x)
                y2 = min(frame.shape[0], y + h + pad_y)
                crop = frame[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)

            now = time.monotonic()
            should_save = (
                auto_mode
                and crop is not None
                and (now - last_save_at) >= args.interval_sec
                and saved < args.target_count
            )
            if should_save:
                img = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
                file_path = out_dir / f"{args.person}_{saved:04d}.jpg"
                cv2.imwrite(str(file_path), img)
                saved += 1
                last_save_at = now

            status = f"{args.label} saved={saved}/{args.target_count} auto={'on' if auto_mode else 'off'}"
            cv2.putText(
                frame,
                status,
                (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Anti-Spoof Capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a"):
                auto_mode = not auto_mode
            if (
                key == ord(" ")
                and crop is not None
                and saved < args.target_count
            ):
                img = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
                file_path = out_dir / f"{args.person}_{saved:04d}.jpg"
                cv2.imwrite(str(file_path), img)
                saved += 1
                last_save_at = now

            if saved >= args.target_count:
                print("Target reached.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. Saved: {saved} images -> {out_dir}")


if __name__ == "__main__":
    main()
