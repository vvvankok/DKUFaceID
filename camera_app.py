from __future__ import annotations

import argparse
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.faceid.modeling import (
    AccessEvent,
    IdentityRecord,
    cleanup_old_events,
    cosine_similarity,
    get_attendance_today_utc,
    load_identities,
    log_access_event,
    mark_recent_false_incidents,
    normalize_vector,
    upsert_identity_embedding,
)
from src.faceid.alerts import TelegramAlerter


def build_parser() -> argparse.ArgumentParser:
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--db-path",
            type=Path,
            default=Path("artifacts/faceid_identities.db"),
            help="SQLite path with identity embeddings.",
        )
        p.add_argument(
            "--camera-index",
            type=int,
            default=0,
            help="OpenCV camera index.",
        )
        p.add_argument(
            "--device",
            type=str,
            default=None,
            help="Embedding device: cpu or cuda.",
        )
        p.add_argument(
            "--weights-path",
            type=Path,
            default=None,
            help="Local FaceNet weights (.pt) to avoid internet download.",
        )

    parser = argparse.ArgumentParser(
        description="Camera registration and verification for FaceID."
    )
    add_common_args(parser)

    sub = parser.add_subparsers(dest="mode", required=True)

    register = sub.add_parser("register", help="Register or update a user from camera.")
    add_common_args(register)
    register.add_argument("--name", type=str, required=True, help="User name/ID.")
    register.add_argument(
        "--samples",
        type=int,
        default=20,
        help="How many face embeddings to capture.",
    )
    register.add_argument(
        "--capture-interval-sec",
        type=float,
        default=0.5,
        help="Minimal interval between captured samples.",
    )

    verify = sub.add_parser("verify", help="Verify user in real-time from camera.")
    add_common_args(verify)
    verify.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Cosine similarity threshold for successful verification.",
    )
    verify.add_argument(
        "--show-score",
        action="store_true",
        help="Overlay similarity score on frame.",
    )
    verify.add_argument(
        "--liveness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable active liveness challenge before identity verification.",
    )
    verify.add_argument(
        "--liveness-mode",
        type=str,
        choices=["passive", "fast", "motion", "blink"],
        default="passive",
        help="Liveness method: passive, fast, motion challenge, or blink challenge.",
    )
    verify.add_argument(
        "--liveness-move-px",
        type=float,
        default=35.0,
        help="Required face-center shift in pixels to pass liveness challenge.",
    )
    verify.add_argument(
        "--liveness-timeout-sec",
        type=float,
        default=6.0,
        help="Timeout for a single liveness challenge.",
    )
    verify.add_argument(
        "--liveness-hold-sec",
        type=float,
        default=3.0,
        help="How long liveness stays valid after passing.",
    )
    verify.add_argument(
        "--blink-target",
        type=int,
        default=1,
        help="How many blinks are required to pass blink liveness.",
    )
    verify.add_argument(
        "--blink-ear-threshold",
        type=float,
        default=0.20,
        help="EAR threshold below which eye is considered closed.",
    )
    verify.add_argument(
        "--blink-min-closed-sec",
        type=float,
        default=0.8,
        help="How long eyes must stay closed for one valid blink.",
    )
    verify.add_argument(
        "--fast-window-sec",
        type=float,
        default=0.7,
        help="Time window for fast liveness analysis.",
    )
    verify.add_argument(
        "--fast-face-jitter-px",
        type=float,
        default=5.0,
        help="Required face center motion (px) inside fast liveness window.",
    )
    verify.add_argument(
        "--fast-ear-delta",
        type=float,
        default=0.015,
        help="Required eye aspect ratio delta in fast liveness window.",
    )
    verify.add_argument(
        "--fast-require-blink",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require at least one real blink in fast liveness mode.",
    )
    verify.add_argument(
        "--fast-blink-closed-threshold",
        type=float,
        default=0.19,
        help="EAR threshold for closed eyes in fast blink detector.",
    )
    verify.add_argument(
        "--fast-blink-open-threshold",
        type=float,
        default=0.24,
        help="EAR threshold for open eyes in fast blink detector.",
    )
    verify.add_argument(
        "--fast-blink-min-closed-frames",
        type=int,
        default=2,
        help="Minimum consecutive closed-eye frames for valid blink in fast mode.",
    )
    verify.add_argument(
        "--fast-blink-max-sec",
        type=float,
        default=1.0,
        help="Maximum closed-eye duration for valid blink in fast mode.",
    )
    verify.add_argument(
        "--passive-window-sec",
        type=float,
        default=0.6,
        help="Time window for passive liveness analysis.",
    )
    verify.add_argument(
        "--passive-residual-flow",
        type=float,
        default=0.18,
        help="Minimum residual optical flow for passive liveness.",
    )
    verify.add_argument(
        "--passive-min-texture",
        type=float,
        default=30.0,
        help="Minimum Laplacian texture variance in face ROI.",
    )
    verify.add_argument(
        "--passive-max-face-jitter-px",
        type=float,
        default=10.0,
        help="Maximum allowed face center jitter for passive attention focus.",
    )
    verify.add_argument(
        "--event-ttl-days",
        type=int,
        default=7,
        help="How many days to keep access events in DB.",
    )
    verify.add_argument(
        "--event-cooldown-sec",
        type=float,
        default=1.2,
        help="Minimum interval between repeated same event logs.",
    )
    verify.add_argument(
        "--telegram-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Telegram alerts for suspicious events.",
    )
    verify.add_argument(
        "--telegram-bot-token",
        type=str,
        default="",
        help="Telegram bot token.",
    )
    verify.add_argument(
        "--telegram-chat-id",
        type=str,
        default="",
        help="Telegram chat id for alerts.",
    )
    verify.add_argument(
        "--telegram-fail-threshold",
        type=int,
        default=3,
        help="How many deny events in window trigger Telegram alert.",
    )
    verify.add_argument(
        "--telegram-window-sec",
        type=float,
        default=20.0,
        help="Time window for deny-event accumulation.",
    )
    verify.add_argument(
        "--telegram-alert-cooldown-sec",
        type=float,
        default=30.0,
        help="Minimum interval between Telegram alerts.",
    )
    verify.add_argument(
        "--telegram-send-photo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send photo with Telegram alerts.",
    )
    verify.add_argument(
        "--telegram-unknown-cooldown-sec",
        type=float,
        default=20.0,
        help="Minimum interval between unknown-person alerts.",
    )
    verify.add_argument(
        "--telegram-unknown-min-presence-sec",
        type=float,
        default=2.5,
        help="Unknown person must stay this long before Telegram alert.",
    )
    verify.add_argument(
        "--telegram-admin-buttons",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show allow/deny inline buttons for unknown person alerts.",
    )
    verify.add_argument(
        "--telegram-admin-allow-sec",
        type=float,
        default=15.0,
        help="How long admin 'allow' action remains active for unknown user.",
    )
    verify.add_argument(
        "--telegram-suspicious-min-presence-sec",
        type=float,
        default=3.5,
        help="Suspicious deny streak must persist this long before Telegram alert.",
    )
    verify.add_argument(
        "--audit-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable false-incident audit and auto-marking.",
    )
    verify.add_argument(
        "--audit-lookback-sec",
        type=int,
        default=20,
        help="Lookback window for false-incident marking after ALLOW.",
    )
    return parser


def draw_box(frame: np.ndarray, box: np.ndarray | None, color: tuple[int, int, int]) -> None:
    if box is None:
        return
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    import cv2

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def largest_box(boxes: np.ndarray | None) -> np.ndarray | None:
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(np.argmax(areas))]


def put_status(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    import cv2

    cv2.putText(
        frame,
        text,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def run_register(args: argparse.Namespace) -> None:
    import cv2
    from src.faceid.embedding import FaceEmbedder

    if args.samples < 1:
        raise ValueError("--samples must be >= 1")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    embedder = FaceEmbedder(device=args.device, weights_path=args.weights_path)
    captured: list[np.ndarray] = []
    last_capture_at = 0.0

    print("Registration mode started. Press 'q' to quit.")
    print(f"Registering user: {args.name}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            pil = Image.fromarray(frame[:, :, ::-1])
            boxes = embedder.detect_boxes(pil)
            box = largest_box(boxes)
            draw_box(frame, box, (255, 255, 0))

            now = time.monotonic()
            has_face = box is not None
            ready = now - last_capture_at >= args.capture_interval_sec
            if has_face and ready and len(captured) < args.samples:
                embedding = embedder.extract_single(pil)
                if embedding is not None:
                    captured.append(embedding)
                    last_capture_at = now

            put_status(
                frame,
                f"Register {args.name}: {len(captured)}/{args.samples}",
                (0, 255, 255),
            )
            cv2.imshow("FaceID Register", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if len(captured) >= args.samples:
                break

        if not captured:
            raise RuntimeError("No face samples captured.")

        centroid = normalize_vector(np.mean(np.stack(captured, axis=0), axis=0))
        upsert_identity_embedding(args.db_path, args.name, centroid)
        print(f"User '{args.name}' saved to {args.db_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def match_identity(
    embedding: np.ndarray, identities: list[IdentityRecord], threshold: float
) -> tuple[str, float]:
    best_name = "unknown"
    best_score = -1.0
    for identity in identities:
        score = cosine_similarity(embedding, identity.embedding)
        if score > best_score:
            best_name = identity.name
            best_score = score
    if best_score < threshold:
        return "unknown", best_score
    return best_name, best_score


@dataclass(frozen=True)
class AccessDecision:
    decision: str
    reason: str
    identity: str
    score: float
    color: tuple[int, int, int]
    display_text: str


def make_decision(
    *,
    has_face: bool,
    liveness_required: bool,
    liveness_ok: bool,
    identity: str,
    score: float,
) -> AccessDecision:
    if not has_face:
        return AccessDecision(
            decision="DENY_NO_FACE",
            reason="no_face",
            identity="no_face",
            score=-1.0,
            color=(0, 0, 255),
            display_text="no_face",
        )
    if liveness_required and not liveness_ok:
        return AccessDecision(
            decision="DENY_LIVENESS",
            reason="liveness_failed",
            identity="unknown",
            score=score,
            color=(0, 165, 255),
            display_text="liveness_required",
        )
    if identity == "unknown":
        return AccessDecision(
            decision="DENY_UNKNOWN",
            reason="not_in_db_or_low_score",
            identity="unknown",
            score=score,
            color=(0, 165, 255),
            display_text="unknown",
        )
    return AccessDecision(
        decision="ALLOW",
        reason="verified",
        identity=identity,
        score=score,
        color=(0, 200, 0),
        display_text=identity,
    )


def face_center(box: np.ndarray) -> tuple[float, float]:
    return float((box[0] + box[2]) * 0.5), float((box[1] + box[3]) * 0.5)


def random_challenge() -> str:
    return random.choice(["left", "right", "up", "down"])


def challenge_text(direction: str) -> str:
    return f"Liveness: move {direction}"


def challenge_passed(
    direction: str,
    baseline_center: tuple[float, float],
    current_center: tuple[float, float],
    threshold_px: float,
) -> bool:
    dx = current_center[0] - baseline_center[0]
    dy = current_center[1] - baseline_center[1]
    if direction == "left":
        return dx <= -threshold_px
    if direction == "right":
        return dx >= threshold_px
    if direction == "up":
        return dy <= -threshold_px
    return dy >= threshold_px


def euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def eye_aspect_ratio(points: list[tuple[float, float]]) -> float:
    # points order: p1, p2, p3, p4, p5, p6
    vertical = euclid(points[1], points[5]) + euclid(points[2], points[4])
    horizontal = 2.0 * euclid(points[0], points[3])
    if horizontal <= 1e-6:
        return 1.0
    return vertical / horizontal


def get_blink_ear(frame: np.ndarray, face_mesh: object) -> float | None:
    rgb = frame[:, :, ::-1]
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]

    def p(i: int) -> tuple[float, float]:
        return (lm[i].x * w, lm[i].y * h)

    left_idx = [33, 160, 158, 133, 153, 144]
    right_idx = [362, 385, 387, 263, 373, 380]
    left_ear = eye_aspect_ratio([p(i) for i in left_idx])
    right_ear = eye_aspect_ratio([p(i) for i in right_idx])
    return float((left_ear + right_ear) * 0.5)


def trim_history(
    history: deque[tuple[float, float]],
    now: float,
    window_sec: float,
) -> None:
    while history and (now - history[0][0]) > window_sec:
        history.popleft()


def trim_center_history(
    history: deque[tuple[float, float, float]],
    now: float,
    window_sec: float,
) -> None:
    while history and (now - history[0][0]) > window_sec:
        history.popleft()


def crop_face_gray(frame: np.ndarray, box: np.ndarray, size: int = 96) -> np.ndarray | None:
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return gray


def passive_flow_metrics(prev_gray: np.ndarray, curr_gray: np.ndarray) -> tuple[float, float]:
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=13,
        iterations=2,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    fx = flow[..., 0]
    fy = flow[..., 1]
    mean_fx = float(np.mean(fx))
    mean_fy = float(np.mean(fy))
    residual = np.sqrt((fx - mean_fx) ** 2 + (fy - mean_fy) ** 2)
    residual_mean = float(np.mean(residual))
    texture = float(cv2.Laplacian(curr_gray, cv2.CV_64F).var())
    return residual_mean, texture


def encode_alert_photo(
    frame: np.ndarray, box: np.ndarray | None, decision_text: str
) -> bytes | None:
    import cv2

    annotated = frame.copy()
    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
    cv2.putText(
        annotated,
        decision_text,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return None
    return bytes(buf)


def start_telegram_callback_worker(
    telegram: TelegramAlerter,
    state: dict[str, float],
    stop_event: threading.Event,
    allow_sec: float,
    db_path: Path,
) -> threading.Thread:
    def attendance_message() -> str:
        rows = get_attendance_today_utc(db_path)
        if not rows:
            return "Attendance today: empty."
        lines = ["Attendance today:"]
        for name, _cnt in rows:
            lines.append(name)
        return "\n".join(lines)

    def worker() -> None:
        offset: int | None = None
        while not stop_event.is_set():
            try:
                payload = telegram.get_updates(
                    offset=offset,
                    timeout=8,
                    allowed_updates=["callback_query", "message"],
                )
                updates = payload.get("result", []) if isinstance(payload, dict) else []
                for upd in updates:
                    update_id = upd.get("update_id")
                    if isinstance(update_id, int):
                        offset = update_id + 1

                    msg = upd.get("message") or {}
                    text = str(msg.get("text", "")).strip().lower()
                    if text in {"/attendance", "attendance", "attendance today"}:
                        telegram.send(attendance_message())

                    cb = upd.get("callback_query") or {}
                    data = str(cb.get("data", ""))
                    cb_id = str(cb.get("id", ""))
                    if data.startswith("faceid:allow"):
                        state["allow_until"] = time.monotonic() + max(1.0, allow_sec)
                        if cb_id:
                            telegram.answer_callback_query(cb_id, "Access temporarily allowed")
                    elif data.startswith("faceid:deny"):
                        state["allow_until"] = 0.0
                        if cb_id:
                            telegram.answer_callback_query(cb_id, "Access denied")
                    elif data.startswith("faceid:attendance"):
                        telegram.send(attendance_message())
                        if cb_id:
                            telegram.answer_callback_query(cb_id, "Attendance sent")
            except Exception:
                time.sleep(1.0)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def run_verify(args: argparse.Namespace) -> None:
    import cv2
    from src.faceid.embedding import FaceEmbedder

    identities = load_identities(args.db_path)
    if not identities:
        raise RuntimeError(
            f"No identities in DB: {args.db_path}. Run registration or training first."
        )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    embedder = FaceEmbedder(device=args.device, weights_path=args.weights_path)
    print("Verification mode started. Press 'q' to quit.")
    if args.event_ttl_days > 0:
        deleted = cleanup_old_events(args.db_path, ttl_days=args.event_ttl_days)
        if deleted > 0:
            print(f"Cleaned old events: {deleted}")

    liveness_direction: str | None = None
    liveness_baseline: tuple[float, float] | None = None
    liveness_started_at = 0.0
    liveness_ok_until = 0.0
    blink_count = 0
    eyes_closed_since: float | None = None
    long_close_ready = False
    fast_centers: deque[tuple[float, float, float]] = deque()
    fast_ears: deque[tuple[float, float]] = deque()
    fast_blink_count = 0
    fast_eye_state = "open"
    fast_closed_frames = 0
    fast_closed_started_at: float | None = None
    passive_prev_gray: np.ndarray | None = None
    passive_residual_hist: deque[tuple[float, float]] = deque()
    passive_texture_hist: deque[tuple[float, float]] = deque()
    passive_center_hist: deque[tuple[float, float, float]] = deque()
    last_logged_key: tuple[str, str, str] | None = None
    last_logged_at = 0.0
    deny_events: deque[float] = deque()
    deny_streak_started_at: float | None = None
    unknown_started_at: float | None = None
    last_telegram_alert_at = 0.0
    last_unknown_alert_at = 0.0
    telegram_stop_event = threading.Event()
    telegram_callback_thread: threading.Thread | None = None
    admin_state: dict[str, float] = {"allow_until": 0.0}

    telegram = TelegramAlerter(
        bot_token=args.telegram_bot_token.strip(),
        chat_id=args.telegram_chat_id.strip(),
    )
    if args.telegram_enabled and not telegram.is_configured():
        print("Telegram enabled but token/chat_id are empty. Alerts disabled.")
        args.telegram_enabled = False
    if args.telegram_enabled and args.telegram_admin_buttons:
        telegram_callback_thread = start_telegram_callback_worker(
            telegram=telegram,
            state=admin_state,
            stop_event=telegram_stop_event,
            allow_sec=args.telegram_admin_allow_sec,
            db_path=args.db_path,
        )
        try:
            telegram.send_with_actions(
                "FaceID bot is active.",
                actions=[("Attendance today", "faceid:attendance")],
            )
        except Exception as exc:
            print(f"Telegram menu send failed: {exc}")

    face_mesh = None
    effective_liveness_mode = args.liveness_mode
    if args.liveness and args.liveness_mode in {"blink", "fast"}:
        try:
            import mediapipe as mp
            if not hasattr(mp, "solutions"):
                raise AttributeError("mediapipe has no attribute 'solutions'")
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:
            print(
                "Blink liveness is unavailable (mediapipe issue). "
                f"Falling back to motion liveness. Details: {exc}"
            )
            effective_liveness_mode = "motion"
    if (
        args.liveness
        and effective_liveness_mode == "fast"
        and args.fast_require_blink
        and face_mesh is None
    ):
        print("Fast blink requirement disabled: face landmarks unavailable.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            pil = Image.fromarray(frame[:, :, ::-1])
            boxes = embedder.detect_boxes(pil)
            box = largest_box(boxes)

            name = "unknown"
            score = -1.0
            color = (0, 0, 255)
            status_suffix = ""
            now = time.monotonic()
            liveness_ok = not args.liveness

            if box is not None:
                live_ready = True
                if args.liveness:
                    live_ready = now <= liveness_ok_until
                    if effective_liveness_mode == "motion":
                        center = face_center(box)
                        if not live_ready:
                            if liveness_direction is None:
                                liveness_direction = random_challenge()
                                liveness_baseline = center
                                liveness_started_at = now
                            else:
                                if (
                                    liveness_baseline is not None
                                    and challenge_passed(
                                        liveness_direction,
                                        liveness_baseline,
                                        center,
                                        args.liveness_move_px,
                                    )
                                ):
                                    liveness_ok_until = now + args.liveness_hold_sec
                                    liveness_direction = None
                                    liveness_baseline = None
                                    live_ready = True
                                elif now - liveness_started_at > args.liveness_timeout_sec:
                                    liveness_direction = random_challenge()
                                    liveness_baseline = center
                                    liveness_started_at = now

                        if not live_ready and liveness_direction is not None:
                            status_suffix = f" | {challenge_text(liveness_direction)}"
                            color = (0, 165, 255)
                    elif effective_liveness_mode == "passive":
                        if not live_ready:
                            face_gray = crop_face_gray(frame, box)
                            if face_gray is None:
                                passive_prev_gray = None
                                passive_residual_hist.clear()
                                passive_texture_hist.clear()
                                passive_center_hist.clear()
                                status_suffix = " | liveness passive: no face ROI"
                                color = (0, 165, 255)
                            else:
                                cx, cy = face_center(box)
                                passive_center_hist.append((now, cx, cy))
                                trim_center_history(passive_center_hist, now, args.passive_window_sec)
                                if passive_prev_gray is not None:
                                    residual, texture = passive_flow_metrics(passive_prev_gray, face_gray)
                                    passive_residual_hist.append((now, residual))
                                    passive_texture_hist.append((now, texture))
                                    trim_history(passive_residual_hist, now, args.passive_window_sec)
                                    trim_history(passive_texture_hist, now, args.passive_window_sec)
                                passive_prev_gray = face_gray

                                if passive_residual_hist:
                                    residual_score = max(v for _, v in passive_residual_hist)
                                else:
                                    residual_score = 0.0
                                if passive_texture_hist:
                                    texture_score = max(v for _, v in passive_texture_hist)
                                else:
                                    texture_score = 0.0
                                if len(passive_center_hist) >= 2:
                                    xs = [c[1] for c in passive_center_hist]
                                    ys = [c[2] for c in passive_center_hist]
                                    face_jitter = float(
                                        ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
                                    )
                                else:
                                    face_jitter = 0.0

                                flow_ok = residual_score >= args.passive_residual_flow
                                texture_ok = texture_score >= args.passive_min_texture
                                attention_ok = face_jitter <= args.passive_max_face_jitter_px
                                if flow_ok and texture_ok and attention_ok:
                                    liveness_ok_until = now + args.liveness_hold_sec
                                    live_ready = True
                                    passive_residual_hist.clear()
                                    passive_texture_hist.clear()
                                    passive_center_hist.clear()
                                else:
                                    status_suffix = (
                                        f" | passive flow={residual_score:.3f}/{args.passive_residual_flow:.3f}"
                                        f" tex={texture_score:.1f}/{args.passive_min_texture:.1f}"
                                        f" jitter={face_jitter:.1f}/{args.passive_max_face_jitter_px:.1f}"
                                    )
                                    color = (0, 165, 255)
                    elif effective_liveness_mode == "fast":
                        if not live_ready:
                            cx, cy = face_center(box)
                            fast_centers.append((now, cx, cy))
                            trim_center_history(fast_centers, now, args.fast_window_sec)

                            ear = None
                            if face_mesh is not None:
                                ear = get_blink_ear(frame, face_mesh)
                            if ear is not None:
                                fast_ears.append((now, ear))
                            trim_history(fast_ears, now, args.fast_window_sec)

                            motion_ok = False
                            if len(fast_centers) >= 2:
                                xs = [item[1] for item in fast_centers]
                                ys = [item[2] for item in fast_centers]
                                motion = float(
                                    ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
                                )
                                motion_ok = motion >= args.fast_face_jitter_px
                            else:
                                motion = 0.0

                            ear_ok = True
                            if face_mesh is not None:
                                ear_ok = False
                                if len(fast_ears) >= 2:
                                    ears = [item[1] for item in fast_ears]
                                    ear_delta = float(max(ears) - min(ears))
                                    ear_ok = ear_delta >= args.fast_ear_delta
                                else:
                                    ear_delta = 0.0
                                # Blink detector with hysteresis to avoid false triggers.
                                if ear is not None:
                                    if ear < args.fast_blink_closed_threshold:
                                        fast_closed_frames += 1
                                        if fast_eye_state == "open":
                                            fast_eye_state = "closed"
                                            fast_closed_started_at = now
                                    elif ear > args.fast_blink_open_threshold:
                                        if fast_eye_state == "closed":
                                            closed_for = (
                                                now - fast_closed_started_at
                                                if fast_closed_started_at is not None
                                                else 0.0
                                            )
                                            if (
                                                fast_closed_frames
                                                >= args.fast_blink_min_closed_frames
                                                and closed_for <= args.fast_blink_max_sec
                                            ):
                                                fast_blink_count += 1
                                        fast_eye_state = "open"
                                        fast_closed_frames = 0
                                        fast_closed_started_at = None
                            else:
                                ear_delta = 0.0

                            blink_ok = (not args.fast_require_blink) or (
                                fast_blink_count >= 1 and face_mesh is not None
                            )

                            if motion_ok and ear_ok and blink_ok:
                                liveness_ok_until = now + args.liveness_hold_sec
                                live_ready = True
                                fast_centers.clear()
                                fast_ears.clear()
                                fast_blink_count = 0
                                fast_eye_state = "open"
                                fast_closed_frames = 0
                                fast_closed_started_at = None
                            else:
                                status_suffix = (
                                    f" | liveness fast m={motion:.1f}/{args.fast_face_jitter_px:.1f}"
                                )
                                if face_mesh is not None:
                                    status_suffix += (
                                        f" e={ear_delta:.3f}/{args.fast_ear_delta:.3f}"
                                    )
                                    if args.fast_require_blink:
                                        status_suffix += f" b={fast_blink_count}/1"
                                color = (0, 165, 255)
                    else:
                        if not live_ready and face_mesh is not None:
                            ear = get_blink_ear(frame, face_mesh)
                            if ear is None:
                                blink_count = 0
                                eyes_closed_since = None
                                long_close_ready = False
                                status_suffix = " | liveness: eyes not visible"
                            else:
                                if ear < args.blink_ear_threshold:
                                    if eyes_closed_since is None:
                                        eyes_closed_since = now
                                    closed_for = now - eyes_closed_since
                                    if closed_for >= args.blink_min_closed_sec:
                                        long_close_ready = True
                                        status_suffix = (
                                            f" | eyes closed {closed_for:.1f}s, now open"
                                        )
                                    else:
                                        status_suffix = (
                                            f" | hold eyes closed {closed_for:.1f}s/"
                                            f"{args.blink_min_closed_sec:.1f}s"
                                        )
                                else:
                                    if long_close_ready:
                                        blink_count += 1
                                    long_close_ready = False
                                    eyes_closed_since = None
                                if blink_count >= args.blink_target:
                                    liveness_ok_until = now + args.liveness_hold_sec
                                    blink_count = 0
                                    eyes_closed_since = None
                                    long_close_ready = False
                                    live_ready = True
                                else:
                                    status_suffix = (
                                        f" | liveness: blink {blink_count}/{args.blink_target}"
                                    )
                                    color = (0, 165, 255)

                if live_ready:
                    liveness_ok = True
                    embedding = embedder.extract_single(pil)
                    if embedding is not None:
                        name, score = match_identity(embedding, identities, args.threshold)
                        color = (0, 200, 0) if name != "unknown" else (0, 165, 255)
                        if args.liveness:
                            status_suffix = " | liveness OK"
            else:
                liveness_direction = None
                liveness_baseline = None
                blink_count = 0
                eyes_closed_since = None
                long_close_ready = False
                passive_prev_gray = None
                passive_residual_hist.clear()
                passive_texture_hist.clear()
                passive_center_hist.clear()
                fast_centers.clear()
                fast_ears.clear()
                fast_blink_count = 0
                fast_eye_state = "open"
                fast_closed_frames = 0
                fast_closed_started_at = None

            decision = make_decision(
                has_face=box is not None,
                liveness_required=args.liveness,
                liveness_ok=liveness_ok,
                identity=name,
                score=score,
            )
            if (
                args.telegram_enabled
                and admin_state.get("allow_until", 0.0) > now
                and decision.decision == "DENY_UNKNOWN"
            ):
                decision = AccessDecision(
                    decision="ALLOW",
                    reason="admin_override",
                    identity="admin_override",
                    score=decision.score,
                    color=(0, 200, 0),
                    display_text="admin_override",
                )
            color = decision.color

            event_key = (decision.decision, decision.reason, decision.identity)
            if (
                event_key != last_logged_key
                or (now - last_logged_at) >= args.event_cooldown_sec
            ):
                log_access_event(
                    args.db_path,
                    AccessEvent(
                        decision=decision.decision,
                        reason=decision.reason,
                        identity=decision.identity,
                        score=decision.score,
                        liveness_ok=liveness_ok,
                        source="camera_verify",
                        is_suspicious=decision.decision != "ALLOW",
                    ),
                )
                last_logged_key = event_key
                last_logged_at = now

                if args.audit_enabled and decision.decision == "ALLOW":
                    fixed = mark_recent_false_incidents(
                        args.db_path,
                        identity=decision.identity,
                        lookback_sec=args.audit_lookback_sec,
                    )
                    if fixed > 0:
                        print(f"audit: marked false incidents={fixed} for {decision.identity}")

            if decision.decision.startswith("DENY_"):
                deny_events.append(now)
                if deny_streak_started_at is None:
                    deny_streak_started_at = now
            else:
                deny_streak_started_at = None
            while deny_events and (now - deny_events[0]) > args.telegram_window_sec:
                deny_events.popleft()

            should_alert = (
                args.telegram_enabled
                and len(deny_events) >= args.telegram_fail_threshold
                and deny_streak_started_at is not None
                and (now - deny_streak_started_at) >= args.telegram_suspicious_min_presence_sec
                and (now - last_telegram_alert_at) >= args.telegram_alert_cooldown_sec
            )

            if decision.decision == "DENY_UNKNOWN":
                if unknown_started_at is None:
                    unknown_started_at = now
            else:
                unknown_started_at = None

            should_alert_unknown = (
                args.telegram_enabled
                and decision.decision == "DENY_UNKNOWN"
                and unknown_started_at is not None
                and (now - unknown_started_at) >= args.telegram_unknown_min_presence_sec
                and (now - last_unknown_alert_at) >= args.telegram_unknown_cooldown_sec
            )

            if should_alert_unknown:
                caption = (
                    "FaceID unknown person detected\n"
                    f"score={decision.score:.3f}\n"
                    f"liveness_ok={liveness_ok}"
                )
                try:
                    actions = [
                        ("Allow access", "faceid:allow"),
                        ("Deny", "faceid:deny"),
                    ]
                    if args.telegram_send_photo:
                        photo = encode_alert_photo(frame, box, "DENY_UNKNOWN")
                        if photo is not None:
                            if args.telegram_admin_buttons:
                                telegram.send_photo_with_actions(
                                    photo, caption=caption, actions=actions
                                )
                            else:
                                telegram.send_photo(photo, caption=caption)
                        else:
                            if args.telegram_admin_buttons:
                                telegram.send_with_actions(caption, actions=actions)
                            else:
                                telegram.send(caption)
                    else:
                        if args.telegram_admin_buttons:
                            telegram.send_with_actions(caption, actions=actions)
                        else:
                            telegram.send(caption)
                    last_unknown_alert_at = now
                except Exception as exc:
                    print(f"Telegram unknown alert failed: {exc}")

            if should_alert:
                try:
                    caption = (
                        "FaceID suspicious activity\n"
                        f"decision={decision.decision}\n"
                        f"identity={decision.identity}\n"
                        f"score={decision.score:.3f}\n"
                        f"liveness_ok={liveness_ok}\n"
                        f"fails_in_window={len(deny_events)}"
                    )
                    if args.telegram_send_photo:
                        photo = encode_alert_photo(frame, box, "SUSPICIOUS_ACTIVITY")
                        if photo is not None:
                            telegram.send_photo(photo, caption=caption)
                        else:
                            telegram.send(caption)
                    else:
                        telegram.send(caption)
                    last_telegram_alert_at = now
                except Exception as exc:
                    print(f"Telegram alert failed: {exc}")

            draw_box(frame, box, color)
            text = decision.display_text
            if args.show_score and decision.score >= 0:
                text = f"{decision.display_text} ({decision.score:.3f})"
            text += status_suffix
            put_status(frame, text, color)

            cv2.imshow("FaceID Verify", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        telegram_stop_event.set()
        if telegram_callback_thread is not None:
            telegram_callback_thread.join(timeout=0.5)
        if face_mesh is not None:
            face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required. Install dependencies: pip install -r requirements.txt"
        ) from exc

    if args.mode == "register":
        run_register(args)
    elif args.mode == "verify":
        run_verify(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
