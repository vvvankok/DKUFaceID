from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.faceid.modeling import (
    AccessEvent,
    IdentityRecord,
    check_schedule_allowed,
    cleanup_old_events,
    cosine_similarity,
    get_attendance_today_utc,
    load_identities,
    log_access_event,
    log_unknown_visit,
    mark_recent_false_incidents,
    normalize_vector,
    save_individual_embeddings,
    upsert_identity_embedding,
    upsert_identity_photo,
)
from src.faceid.alerts import MultiTelegramAlerter, TelegramAlerter
from src.faceid.relay import RelayController


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
        "--anti-spoof",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable pretrained anti-spoof model before FaceID matching.",
    )
    verify.add_argument(
        "--anti-spoof-model-path",
        type=Path,
        default=Path("artifacts/anti_spoof_model.pt"),
        help="Path to pretrained anti-spoof TorchScript model (.pt).",
    )
    verify.add_argument(
        "--anti-spoof-threshold",
        type=float,
        default=0.80,
        help="Live confidence threshold for anti-spoof decision.",
    )
    verify.add_argument(
        "--anti-spoof-input-size",
        type=int,
        default=128,
        help="Input size for anti-spoof model preprocessing.",
    )
    verify.add_argument(
        "--anti-spoof-margin-ratio",
        type=float,
        default=0.20,
        help="Extra crop margin around detected face for anti-spoof inference.",
    )
    verify.add_argument(
        "--anti-spoof-min-interval-sec",
        type=float,
        default=0.10,
        help="Minimum interval between anti-spoof inferences for CPU efficiency.",
    )
    verify.add_argument(
        "--liveness-hold-sec",
        type=float,
        default=3.0,
        help="How long liveness stays valid after passing.",
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
        help="Telegram chat id for alerts (recipient 1).",
    )
    verify.add_argument(
        "--telegram-chat-id2",
        type=str,
        default="",
        help="Second Telegram chat id for alerts (recipient 2, optional).",
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
        default=40.0,
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
    verify.add_argument(
        "--relay-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Activate relay/door-lock on successful access.",
    )
    verify.add_argument(
        "--relay-pin",
        type=int,
        default=18,
        help="BCM GPIO pin number for the relay (Raspberry Pi only).",
    )
    verify.add_argument(
        "--relay-duration-sec",
        type=float,
        default=2.0,
        help="How many seconds the relay stays open on ALLOW.",
    )
    verify.add_argument(
        "--relay-active-high",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Relay is active HIGH (default) or active LOW.",
    )
    verify.add_argument(
        "--schedule-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deny access outside each employee's configured schedule.",
    )
    verify.add_argument(
        "--cluster-unknowns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log unknown visitor embeddings for later clustering analysis.",
    )
    verify.add_argument(
        "--cluster-unknowns-cooldown-sec",
        type=float,
        default=5.0,
        help="Min interval between unknown visit log entries.",
    )
    verify.add_argument(
        "--threshold-after-hours",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use stricter cosine threshold outside working hours.",
    )
    verify.add_argument(
        "--threshold-night",
        type=float,
        default=0.82,
        help="Stricter cosine threshold used outside working hours.",
    )
    verify.add_argument(
        "--threshold-night-start",
        type=int,
        default=20,
        help="Hour (local, 0-23) when stricter threshold kicks in.",
    )
    verify.add_argument(
        "--threshold-night-end",
        type=int,
        default=8,
        help="Hour (local, 0-23) when normal threshold resumes.",
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
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def largest_box(boxes: np.ndarray | None) -> np.ndarray | None:
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return boxes[int(np.argmax(areas))]


def get_effective_threshold(args: argparse.Namespace) -> float:
    """Return stricter threshold outside working hours if feature is enabled."""
    if not getattr(args, "threshold_after_hours", False):
        return args.threshold
    import datetime
    hour = datetime.datetime.now().hour
    start = args.threshold_night_start
    end = args.threshold_night_end
    # Night window wraps midnight: start > end means e.g. 20–8
    if start > end:
        is_night = (hour >= start) or (hour < end)
    else:
        is_night = start <= hour < end
    return args.threshold_night if is_night else args.threshold


def put_status(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
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
    from src.faceid.embedding import FaceEmbedder

    if args.samples < 1:
        raise ValueError("--samples must be >= 1")

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    embedder = FaceEmbedder(device=args.device, weights_path=args.weights_path)
    captured: list[np.ndarray] = []
    best_face_crop: np.ndarray | None = None
    best_sharpness: float = -1.0
    last_capture_at = 0.0

    print("Registration mode started. Press 'q' to quit.")
    print(f"Registering user: {args.name}")

    _cam_fail = 0
    _CAM_FAIL_MAX = 30

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                _cam_fail += 1
                if _cam_fail >= _CAM_FAIL_MAX:
                    print(f"[camera] read failed {_cam_fail}x, reconnecting index {args.camera_index}...")
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(args.camera_index)
                    _cam_fail = 0
                    if not cap.isOpened():
                        print("[camera] reconnect failed, will retry...")
                continue
            _cam_fail = 0

            pil = Image.fromarray(frame[:, :, ::-1])
            boxes = embedder.detect_boxes(pil)
            box = largest_box(boxes)
            draw_box(frame, box, (255, 255, 0))

            now = time.monotonic()
            has_face = box is not None
            ready = now - last_capture_at >= args.capture_interval_sec
            quality_note = ""
            if has_face and ready and len(captured) < args.samples:
                embedding, quality = embedder.extract_single_gated(pil)
                if embedding is not None:
                    captured.append(embedding)
                    last_capture_at = now
                    # Track best-quality face crop for photo storage
                    if quality > best_sharpness and box is not None:
                        x1, y1, x2, y2 = [int(v) for v in box]
                        h, w = frame.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            best_face_crop = crop.copy()
                            best_sharpness = quality
                elif quality > 0:
                    quality_note = f" LOW SHARPNESS({quality:.0f})"

            status_color = (0, 255, 255) if not quality_note else (0, 165, 255)
            put_status(
                frame,
                f"Register {args.name}: {len(captured)}/{args.samples}{quality_note}",
                status_color,
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
        save_individual_embeddings(args.db_path, args.name, captured)

        # Save best face crop as employee photo
        photo_path = ""
        if best_face_crop is not None:
            photos_dir = Path("artifacts/photos")
            photos_dir.mkdir(parents=True, exist_ok=True)
            photo_file = photos_dir / f"{args.name}.jpg"
            cv2.imwrite(str(photo_file), best_face_crop)
            photo_path = str(photo_file)
            upsert_identity_photo(args.db_path, args.name, photo_path)
            print(f"Photo saved: {photo_file}")

        print(f"User '{args.name}' saved to {args.db_path} ({len(captured)} embeddings)")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def match_identity(
    embedding: np.ndarray, identities: list[IdentityRecord], threshold: float
) -> tuple[str, float]:
    if not identities:
        return "unknown", -1.0
    best_score = -1.0
    best_name = "unknown"
    for record in identities:
        # Use individual embeddings when available, else fall back to centroid
        vecs = record.all_embeddings if record.all_embeddings else (record.embedding,)
        matrix = np.stack(vecs, axis=0)  # (K, D)
        scores = matrix @ embedding       # cosine sim for unit vectors
        score = float(np.max(scores))
        if score > best_score:
            best_score = score
            best_name = record.name
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
    schedule_ok: bool = True,
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
    if not schedule_ok:
        return AccessDecision(
            decision="DENY_SCHEDULE",
            reason="outside_schedule",
            identity=identity,
            score=score,
            color=(100, 0, 200),
            display_text=f"{identity} [вне расписания]",
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


def trim_history(
    history: deque,
    now: float,
    window_sec: float,
) -> None:
    while history and (now - history[0][0]) > window_sec:
        history.popleft()


def crop_face_gray(frame: np.ndarray, box: np.ndarray, size: int = 96) -> np.ndarray | None:
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
    import datetime

    def attendance_html() -> str:
        rows = get_attendance_today_utc(db_path)
        today = datetime.datetime.now().strftime("%d.%m.%Y")
        if not rows:
            return f"<b>Посещаемость {today}</b>\n\nНикто не приходил."
        lines = [f"<b>Посещаемость за {today}</b>  ({len(rows)} чел.)\n"]
        for name, first_time in rows:
            lines.append(f"  &#9989; <b>{name}</b> — вошёл в {first_time} UTC")
        return "\n".join(lines)

    def stats_html() -> str:
        if not db_path.exists():
            return "<b>Статистика</b>\n\nБаза данных не найдена."
        import sqlite3 as _sq
        conn = _sq.connect(str(db_path))
        try:
            today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
            row = conn.execute(
                """SELECT
                    SUM(decision='ALLOW') AS ok,
                    SUM(decision LIKE 'DENY%') AS denied,
                    SUM(is_suspicious) AS sus
                   FROM access_events WHERE ts_utc >= ?""",
                (today,),
            ).fetchone()
        finally:
            conn.close()
        ok, denied, sus = (int(v or 0) for v in (row or (0, 0, 0)))
        total = ok + denied
        return (
            f"<b>Статистика за сегодня</b>\n\n"
            f"&#9989; Разрешено: <b>{ok}</b>\n"
            f"&#10060; Отказов: <b>{denied}</b>\n"
            f"&#128683; Подозрительных: <b>{sus}</b>\n"
            f"&#128202; Всего событий: <b>{total}</b>"
        )

    def users_html() -> str:
        if not db_path.exists():
            return "<b>Пользователи</b>\n\nБаза данных не найдена."
        import sqlite3 as _sq
        conn = _sq.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT name, updated_at FROM identities ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return "<b>Зарегистрированные пользователи</b>\n\nСписок пуст."
        lines = [f"<b>Зарегистрированные пользователи</b> ({len(rows)}):\n"]
        for name, updated_at in rows:
            date = str(updated_at)[:10] if updated_at else "—"
            lines.append(f"  &#128100; <b>{name}</b>  <i>({date})</i>")
        return "\n".join(lines)

    HELP_TEXT = (
        "<b>FaceID — доступные команды</b>\n\n"
        "/help — эта справка\n"
        "/list — список зарегистрированных пользователей\n"
        "/stats — статистика за сегодня\n"
        "/attendance — посещаемость за сегодня\n"
        "/status — статус системы\n"
        "/report — отчёт за сегодня (PDF)\n"
        "/report ГГГГ-ММ-ДД — отчёт за один день\n"
        "/report ГГГГ-ММ-ДД ГГГГ-ММ-ДД — отчёт за период"
    )

    def _handle_report_cmd(raw_text: str, tg, db: Path) -> None:
        """Parse /report [start] [end] and send PDF to all recipients."""
        import tempfile
        from src.faceid.modeling import export_pdf_report
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        parts = raw_text.strip().split()
        # /report                    → today
        # /report ГГГГ-ММ-ДД        → single day
        # /report ГГГГ-ММ-ДД ГГГГ-ММ-ДД → range
        if len(parts) == 1:
            start_date = end_date = today
        elif len(parts) == 2:
            start_date = end_date = parts[1]
        elif len(parts) >= 3:
            start_date, end_date = parts[1], parts[2]
        else:
            start_date = end_date = today

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            n = export_pdf_report(db, start_date, end_date, tmp_path)
            pdf_bytes = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)
            filename = f"faceid_report_{start_date}_{end_date}.pdf"
            caption = (
                f"&#128196; <b>FaceID — Отчёт о посещаемости</b>\n"
                f"Период: <code>{start_date}</code> — <code>{end_date}</code>\n"
                f"Строк: <b>{n}</b>"
            )
            tg.send_document(pdf_bytes, filename, caption=caption, parse_mode="HTML")
        except Exception as exc:
            tg.send_html(f"&#10060; Ошибка при формировании отчёта:\n<code>{exc}</code>")

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
                    raw_text = str(msg.get("text", "")).strip()
                    cmd = raw_text.lower().split("@")[0]  # strip @botname suffix

                    if cmd in {"/help", "/команды", "help"}:
                        telegram.send_html(HELP_TEXT)
                    elif cmd in {"/list", "/пользователи", "/users"}:
                        telegram.send_html(users_html())
                    elif cmd in {"/stats", "/статистика", "/stat"}:
                        telegram.send_html(stats_html())
                    elif cmd in {"/attendance", "/посещаемость", "attendance"}:
                        telegram.send_html(attendance_html())
                    elif cmd in {"/status", "/статус"}:
                        now_str = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                        telegram.send_html(
                            f"<b>Статус системы FaceID</b>\n\n"
                            f"&#128994; Система активна\n"
                            f"&#128336; Время: <code>{now_str}</code>"
                        )
                    elif raw_text.lower().startswith("/report"):
                        _handle_report_cmd(raw_text, telegram, db_path)

                    cb = upd.get("callback_query") or {}
                    data = str(cb.get("data", ""))
                    cb_id = str(cb.get("id", ""))
                    if data.startswith("faceid:allow"):
                        state["allow_until"] = time.monotonic() + max(1.0, allow_sec)
                        if cb_id:
                            telegram.answer_callback_query(cb_id, "Доступ временно разрешён")
                    elif data.startswith("faceid:deny"):
                        state["allow_until"] = 0.0
                        if cb_id:
                            telegram.answer_callback_query(cb_id, "Доступ заблокирован")
                    elif data.startswith("faceid:attendance"):
                        telegram.send_html(attendance_html())
                        if cb_id:
                            telegram.answer_callback_query(cb_id, "Посещаемость отправлена")
            except Exception:
                time.sleep(1.0)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def run_verify(args: argparse.Namespace) -> None:
    from anti_spoof import AntiSpoofModel
    from src.faceid.embedding import FaceEmbedder

    identities = load_identities(args.db_path)
    if not identities:
        raise RuntimeError(
            f"No identities in DB: {args.db_path}. Run registration or training first."
        )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    # Relay / door-lock controller (ТЗ п. 4.1)
    relay: RelayController | None = None
    if args.relay_enabled:
        relay = RelayController(
            pin=args.relay_pin,
            active_high=args.relay_active_high,
            door_open_sec=args.relay_duration_sec,
        )

    embedder = FaceEmbedder(device=args.device, weights_path=args.weights_path)
    anti_spoof: AntiSpoofModel | None = None
    if args.anti_spoof:
        anti_spoof_device = args.device if args.device else "cpu"
        try:
            anti_spoof = AntiSpoofModel(
                model_path=args.anti_spoof_model_path,
                threshold=args.anti_spoof_threshold,
                device=anti_spoof_device,
                input_size=args.anti_spoof_input_size,
                margin_ratio=args.anti_spoof_margin_ratio,
                min_interval_sec=args.anti_spoof_min_interval_sec,
            )
        except Exception as exc:
            print(
                "Anti-spoof initialization failed, fallback to built-in liveness. "
                "Check --anti-spoof-model-path and model format (.pt TorchScript). "
                f"Details: {exc}"
            )
            args.anti_spoof = False
        else:
            print(
                "Anti-spoof enabled: "
                f"model={args.anti_spoof_model_path} "
                f"threshold={args.anti_spoof_threshold:.2f}"
            )
    print("Verification mode started. Press 'q' to quit.")
    if args.event_ttl_days > 0:
        deleted = cleanup_old_events(args.db_path, ttl_days=args.event_ttl_days)
        if deleted > 0:
            print(f"Cleaned old events: {deleted}")

    liveness_ok_until = 0.0
    passive_prev_gray: np.ndarray | None = None
    passive_residual_hist: deque[tuple[float, float]] = deque()
    passive_texture_hist: deque[tuple[float, float]] = deque()
    passive_center_hist: deque[tuple[float, float, float]] = deque()
    # ── temporal embedding buffer (smoothing, anti-flicker) ──────────────────
    # Keep the last N high-quality embeddings and match against their mean.
    # This is the primary reason phones are more stable than single-frame systems.
    _EMBED_BUF_SIZE = 6
    embed_buffer: deque[np.ndarray] = deque(maxlen=_EMBED_BUF_SIZE)

    last_logged_key: tuple[str, str, str] | None = None
    last_logged_at = 0.0
    last_unknown_visit_at = 0.0
    deny_events: deque[float] = deque()
    deny_streak_started_at: float | None = None
    unknown_started_at: float | None = None
    last_telegram_alert_at = 0.0
    last_unknown_alert_at = 0.0
    telegram_stop_event = threading.Event()
    telegram_callback_thread: threading.Thread | None = None
    admin_state: dict[str, float] = {"allow_until": 0.0}

    _token = args.telegram_bot_token.strip()
    _alerters = [TelegramAlerter(bot_token=_token, chat_id=args.telegram_chat_id.strip())]
    if getattr(args, "telegram_chat_id2", "").strip():
        _alerters.append(TelegramAlerter(bot_token=_token, chat_id=args.telegram_chat_id2.strip()))
        print(f"Telegram: broadcasting to {len(_alerters)} recipients.")
    telegram = MultiTelegramAlerter(alerters=_alerters)
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

    _cam_fail = 0
    _CAM_FAIL_MAX = 30

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                _cam_fail += 1
                if _cam_fail >= _CAM_FAIL_MAX:
                    print(f"[camera] read failed {_cam_fail}x, reconnecting index {args.camera_index}...")
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(args.camera_index)
                    _cam_fail = 0
                    if not cap.isOpened():
                        print("[camera] reconnect failed, will retry...")
                continue
            _cam_fail = 0

            pil = Image.fromarray(frame[:, :, ::-1])
            boxes = embedder.detect_boxes(pil)
            box = largest_box(boxes)

            name = "unknown"
            score = -1.0
            color = (0, 0, 255)
            status_suffix = ""
            now = time.monotonic()
            liveness_required = args.liveness or args.anti_spoof
            liveness_ok = not liveness_required

            if box is not None:
                live_ready = True
                anti_spoof_blocked = False
                if args.anti_spoof and anti_spoof is not None:
                    try:
                        anti_spoof_result = anti_spoof.predict(frame, box, now)
                        if not anti_spoof_result.is_live:
                            live_ready = False
                            anti_spoof_blocked = True
                            status_suffix = (
                                " | anti-spoof: FAKE "
                                f"{anti_spoof_result.live_score:.2f}/"
                                f"{args.anti_spoof_threshold:.2f}"
                            )
                            color = (0, 165, 255)
                    except Exception as exc:
                        live_ready = False
                        anti_spoof_blocked = True
                        status_suffix = f" | anti-spoof error: {exc}"
                        color = (0, 165, 255)

                if live_ready and args.liveness:
                    live_ready = now <= liveness_ok_until
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
                            trim_history(passive_center_hist, now, args.passive_window_sec)
                            if passive_prev_gray is not None:
                                residual, texture = passive_flow_metrics(passive_prev_gray, face_gray)
                                passive_residual_hist.append((now, residual))
                                passive_texture_hist.append((now, texture))
                                trim_history(passive_residual_hist, now, args.passive_window_sec)
                                trim_history(passive_texture_hist, now, args.passive_window_sec)
                            passive_prev_gray = face_gray

                            residual_score = max((v for _, v in passive_residual_hist), default=0.0)
                            texture_score = max((v for _, v in passive_texture_hist), default=0.0)
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

                if live_ready:
                    liveness_ok = True
                    embedding, quality = embedder.extract_single_gated(pil)
                    if embedding is not None:
                        embed_buffer.append(embedding)
                        # Use mean of buffered embeddings — eliminates single-frame
                        # noise, mirrors the temporal smoothing used on phones.
                        if len(embed_buffer) >= 2:
                            mean_emb = normalize_vector(
                                np.mean(np.stack(list(embed_buffer), axis=0), axis=0)
                            )
                        else:
                            mean_emb = embedding
                        name, score = match_identity(mean_emb, identities, get_effective_threshold(args))
                        # Log unknown visit for clustering analysis
                        if name == "unknown" and args.cluster_unknowns:
                            if now - last_unknown_visit_at >= args.cluster_unknowns_cooldown_sec:
                                log_unknown_visit(args.db_path, mean_emb)
                                last_unknown_visit_at = now
                        color = (0, 200, 0) if name != "unknown" else (0, 165, 255)
                        ok_suffix = []
                        if args.anti_spoof:
                            ok_suffix.append("anti-spoof OK")
                        if args.liveness:
                            ok_suffix.append("liveness OK")
                        if ok_suffix:
                            status_suffix = " | " + " + ".join(ok_suffix)
                elif anti_spoof_blocked:
                    passive_prev_gray = None
                    passive_residual_hist.clear()
                    passive_texture_hist.clear()
                    passive_center_hist.clear()
            else:
                passive_prev_gray = None
                passive_residual_hist.clear()
                passive_texture_hist.clear()
                passive_center_hist.clear()
                embed_buffer.clear()  # reset temporal buffer when face leaves frame

            # Schedule check for known identities
            sched_ok = True
            if args.schedule_enabled and name != "unknown":
                sched_ok, _ = check_schedule_allowed(args.db_path, name)

            decision = make_decision(
                has_face=box is not None,
                liveness_required=liveness_required,
                liveness_ok=liveness_ok,
                identity=name,
                score=score,
                schedule_ok=sched_ok,
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

            # Activate relay on successful identification (ТЗ п. 4.1)
            if decision.decision == "ALLOW" and relay is not None:
                relay.open_door()

            if should_alert_unknown:
                import datetime as _dt
                ts = _dt.datetime.utcnow().strftime("%H:%M:%S UTC")
                presence = f"{now - (unknown_started_at or now):.1f}"
                caption = (
                    f"&#128683; <b>Неизвестный пользователь</b>\n\n"
                    f"&#8987; Присутствует: <b>{presence}с</b>\n"
                    f"&#128337; <code>{ts}</code>\n"
                    f"&#128270; Схожесть: <code>{decision.score:.3f}</code>\n"
                    f"&#128065; Живость: {'OK' if liveness_ok else 'нет'}"
                )
                try:
                    actions = [
                        ("✅ Разрешить доступ", "faceid:allow"),
                        ("⛔ Отклонить", "faceid:deny"),
                    ]
                    if args.telegram_send_photo:
                        photo = encode_alert_photo(frame, box, "DENY_UNKNOWN")
                        if photo is not None:
                            if args.telegram_admin_buttons:
                                telegram.send_photo_with_actions(
                                    photo, caption=caption, actions=actions, parse_mode="HTML"
                                )
                            else:
                                telegram.send_photo(photo, caption=caption, parse_mode="HTML")
                        else:
                            if args.telegram_admin_buttons:
                                telegram.send_with_actions(
                                    caption, actions=actions, parse_mode="HTML"
                                )
                            else:
                                telegram.send_html(caption)
                    else:
                        if args.telegram_admin_buttons:
                            telegram.send_with_actions(
                                caption, actions=actions, parse_mode="HTML"
                            )
                        else:
                            telegram.send_html(caption)
                    last_unknown_alert_at = now
                except Exception as exc:
                    print(f"Telegram unknown alert failed: {exc}")

            if should_alert:
                import datetime as _dt
                ts = _dt.datetime.utcnow().strftime("%H:%M:%S UTC")
                caption = (
                    f"&#9888; <b>Подозрительная активность</b>\n\n"
                    f"&#128683; Решение: <code>{decision.decision}</code>\n"
                    f"&#128100; Идентификатор: <code>{decision.identity}</code>\n"
                    f"&#128270; Схожесть: <code>{decision.score:.3f}</code>\n"
                    f"&#128337; <code>{ts}</code>\n"
                    f"&#128202; Отказов в окне: <b>{len(deny_events)}</b>"
                )
                try:
                    if args.telegram_send_photo:
                        photo = encode_alert_photo(frame, box, "SUSPICIOUS_ACTIVITY")
                        if photo is not None:
                            telegram.send_photo(photo, caption=caption, parse_mode="HTML")
                        else:
                            telegram.send_html(caption)
                    else:
                        telegram.send_html(caption)
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
        if relay is not None:
            relay.cleanup()
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "register":
        run_register(args)
    elif args.mode == "verify":
        run_verify(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
