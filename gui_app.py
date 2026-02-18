from __future__ import annotations

import sqlite3
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.faceid.alerts import TelegramAlerter


class FaceIdGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FaceID Access Console")
        self.geometry("980x720")
        self.minsize(920, 640)
        self.configure(bg="#f2f4f7")

        self.proc: subprocess.Popen[str] | None = None
        self.output_thread: threading.Thread | None = None

        self.name_var = tk.StringVar(value="artem")
        self.samples_var = tk.StringVar(value="20")
        self.threshold_var = tk.StringVar(value="0.72")
        self.db_path_var = tk.StringVar(value="artifacts/faceid_identities.db")
        self.weights_path_var = tk.StringVar(value="")
        self.camera_index_var = tk.StringVar(value="0")
        self.device_var = tk.StringVar(value="")

        self.liveness_mode_var = tk.StringVar(value="passive")
        self.liveness_enabled_var = tk.BooleanVar(value=True)
        self.show_score_var = tk.BooleanVar(value=True)
        self.event_ttl_days_var = tk.StringVar(value="7")
        self.event_cooldown_var = tk.StringVar(value="1.2")

        self.fast_window_var = tk.StringVar(value="0.8")
        self.fast_face_jitter_var = tk.StringVar(value="6")
        self.fast_ear_delta_var = tk.StringVar(value="0.018")
        self.fast_require_blink_var = tk.BooleanVar(value=True)
        self.passive_residual_var = tk.StringVar(value="0.24")
        self.passive_texture_var = tk.StringVar(value="45")
        self.passive_max_jitter_var = tk.StringVar(value="8")
        self.telegram_enabled_var = tk.BooleanVar(value=False)
        self.telegram_bot_token_var = tk.StringVar(value="")
        self.telegram_chat_id_var = tk.StringVar(value="")
        self.telegram_fail_threshold_var = tk.StringVar(value="3")
        self.telegram_window_var = tk.StringVar(value="20")
        self.telegram_alert_cooldown_var = tk.StringVar(value="30")
        self.telegram_unknown_cooldown_var = tk.StringVar(value="20")
        self.telegram_send_photo_var = tk.BooleanVar(value=True)
        self.telegram_admin_buttons_var = tk.BooleanVar(value=True)
        self.telegram_admin_allow_sec_var = tk.StringVar(value="15")

        self.status_var = tk.StringVar(value="Ready")
        self.mode_var = tk.StringVar(value="Idle")
        self.events_refresh_ms = 1000

        self._setup_style()
        self._build_ui()
        self._refresh_events()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Header.TLabel", background="#f2f4f7", foreground="#111827")
        style.configure("Title.TLabel", background="#f2f4f7", foreground="#111827")
        style.configure("SubTitle.TLabel", background="#f2f4f7", foreground="#4b5563")
        style.configure("Status.TLabel", background="#111827", foreground="#ffffff")
        style.configure("Field.TLabel", background="#ffffff", foreground="#374151")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            header,
            text="FaceID Verification Console",
            style="Title.TLabel",
            font=("Segoe UI", 17, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text="Registration and real-time access verification",
            style="SubTitle.TLabel",
            font=("Segoe UI", 10),
        ).pack(anchor="w", pady=(2, 0))

        top = ttk.Frame(root, style="Card.TFrame", padding=12)
        top.pack(fill=tk.X)

        self._build_common_section(top)
        self._build_actions(top)

        middle = ttk.Frame(root)
        middle.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        middle.grid_columnconfigure(0, weight=1)
        middle.grid_columnconfigure(1, weight=1)
        middle.grid_rowconfigure(0, weight=1)

        reg_card = ttk.LabelFrame(middle, text="Registration", padding=12)
        reg_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._build_register_section(reg_card)

        verify_card = ttk.LabelFrame(middle, text="Verification", padding=12)
        verify_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._build_verify_section(verify_card)

        status_bar = ttk.Frame(root)
        status_bar.pack(fill=tk.X, pady=(10, 6))
        ttk.Label(
            status_bar,
            textvariable=self.mode_var,
            style="Status.TLabel",
            padding=(10, 4),
        ).pack(side=tk.LEFT)
        ttk.Label(status_bar, textvariable=self.status_var).pack(side=tk.LEFT, padx=(10, 0))

        bottom = ttk.Frame(root)
        bottom.pack(fill=tk.BOTH, expand=True)
        bottom.grid_columnconfigure(0, weight=3)
        bottom.grid_columnconfigure(1, weight=2)
        bottom.grid_rowconfigure(0, weight=1)

        log_card = ttk.LabelFrame(bottom, text="Live Log", padding=8)
        log_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.log = tk.Text(
            log_card,
            height=12,
            wrap=tk.WORD,
            bg="#0f172a",
            fg="#e5e7eb",
            insertbackground="#e5e7eb",
            relief=tk.FLAT,
            font=("Consolas", 10),
        )
        scroll = ttk.Scrollbar(log_card, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set, state=tk.DISABLED)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        events_card = ttk.LabelFrame(bottom, text="Recent Events", padding=8)
        events_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        events_card.grid_columnconfigure(0, weight=1)
        events_card.grid_rowconfigure(0, weight=1)

        columns = ("time", "decision", "identity", "score")
        self.events_tree = ttk.Treeview(events_card, columns=columns, show="headings", height=12)
        self.events_tree.heading("time", text="Time")
        self.events_tree.heading("decision", text="Decision")
        self.events_tree.heading("identity", text="Identity")
        self.events_tree.heading("score", text="Score")
        self.events_tree.column("time", width=86, anchor="w")
        self.events_tree.column("decision", width=96, anchor="w")
        self.events_tree.column("identity", width=90, anchor="w")
        self.events_tree.column("score", width=62, anchor="e")
        self.events_tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll = ttk.Scrollbar(events_card, orient=tk.VERTICAL, command=self.events_tree.yview)
        self.events_tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.grid(row=0, column=1, sticky="ns")

    def _build_common_section(self, parent: ttk.Frame) -> None:
        common = ttk.LabelFrame(parent, text="Environment", padding=10)
        common.pack(fill=tk.X)
        common.grid_columnconfigure(1, weight=1)

        self._field(common, 0, "Camera Index", self.camera_index_var)
        self._field(common, 1, "Database Path", self.db_path_var)
        self._field(common, 2, "Weights Path (.pt)", self.weights_path_var, browse=True)
        self._field(common, 3, "Device (cpu/cuda)", self.device_var)

    def _build_actions(self, parent: ttk.Frame) -> None:
        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(actions, text="Start Registration", command=self.start_register).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(actions, text="Start Verification", command=self.start_verify).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(actions, text="Download Weights", command=self.download_weights).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(actions, text="Test Telegram", command=self.test_telegram).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(actions, text="Stop", command=self.stop_process).pack(side=tk.LEFT)

    def _build_register_section(self, parent: ttk.LabelFrame) -> None:
        parent.grid_columnconfigure(1, weight=1)
        self._field(parent, 0, "User Name", self.name_var)
        self._field(parent, 1, "Samples", self.samples_var)
        ttk.Label(
            parent,
            text="Tip: 20-30 samples per user gives better stability.",
            foreground="#4b5563",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _build_verify_section(self, parent: ttk.LabelFrame) -> None:
        parent.grid_columnconfigure(1, weight=1)
        self._field(parent, 0, "Threshold", self.threshold_var)
        self._field(parent, 1, "Event TTL Days", self.event_ttl_days_var)
        self._field(parent, 2, "Event Cooldown (sec)", self.event_cooldown_var)

        ttk.Label(parent, text="Liveness Mode").grid(row=3, column=0, sticky="w", pady=4)
        mode = ttk.Combobox(
            parent,
            textvariable=self.liveness_mode_var,
            values=["passive", "fast", "blink", "motion"],
            state="readonly",
        )
        mode.grid(row=3, column=1, sticky="ew", pady=4)

        ttk.Checkbutton(parent, text="Enable Liveness", variable=self.liveness_enabled_var).grid(
            row=4, column=0, sticky="w", pady=4
        )
        ttk.Checkbutton(parent, text="Show Score", variable=self.show_score_var).grid(
            row=4, column=1, sticky="w", pady=4
        )

        fast_frame = ttk.LabelFrame(parent, text="Fast Anti-Spoof", padding=8)
        fast_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        fast_frame.grid_columnconfigure(1, weight=1)
        self._field(fast_frame, 0, "Window (sec)", self.fast_window_var)
        self._field(fast_frame, 1, "Face Jitter (px)", self.fast_face_jitter_var)
        self._field(fast_frame, 2, "EAR Delta", self.fast_ear_delta_var)
        ttk.Checkbutton(
            fast_frame,
            text="Require Blink (recommended)",
            variable=self.fast_require_blink_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        passive_frame = ttk.LabelFrame(parent, text="Passive Anti-Spoof", padding=8)
        passive_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        passive_frame.grid_columnconfigure(1, weight=1)
        self._field(passive_frame, 0, "Residual Flow", self.passive_residual_var)
        self._field(passive_frame, 1, "Min Texture", self.passive_texture_var)
        self._field(passive_frame, 2, "Max Face Jitter (px)", self.passive_max_jitter_var)

        tg_frame = ttk.LabelFrame(parent, text="Telegram Alerts", padding=8)
        tg_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tg_frame.grid_columnconfigure(1, weight=1)
        ttk.Checkbutton(tg_frame, text="Enable Telegram Alerts", variable=self.telegram_enabled_var).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
        )
        ttk.Checkbutton(
            tg_frame,
            text="Send Photo with Alert",
            variable=self.telegram_send_photo_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Checkbutton(
            tg_frame,
            text="Admin Buttons (Allow/Deny)",
            variable=self.telegram_admin_buttons_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self._field(tg_frame, 3, "Bot Token", self.telegram_bot_token_var)
        self._field(tg_frame, 4, "Chat ID", self.telegram_chat_id_var)
        self._field(tg_frame, 5, "Fail Threshold", self.telegram_fail_threshold_var)
        self._field(tg_frame, 6, "Window (sec)", self.telegram_window_var)
        self._field(tg_frame, 7, "Alert Cooldown (sec)", self.telegram_alert_cooldown_var)
        self._field(tg_frame, 8, "Unknown Cooldown (sec)", self.telegram_unknown_cooldown_var)
        self._field(tg_frame, 9, "Admin Allow (sec)", self.telegram_admin_allow_sec_var)

    def _field(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.StringVar,
        browse: bool = False,
    ) -> None:
        ttk.Label(parent, text=label, style="Field.TLabel").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        if browse:
            ttk.Button(parent, text="Browse", command=lambda: self._pick_file(var)).grid(
                row=row, column=2, sticky="ew", padx=(8, 0), pady=4
            )

    def _pick_file(self, target_var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(
            title="Select Weights File",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            target_var.set(path)

    def append_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def set_status(self, mode: str, text: str) -> None:
        self.mode_var.set(mode)
        self.status_var.set(text)

    def _validate_common(self) -> tuple[int, Path, Path | None, str]:
        try:
            camera_index = int(self.camera_index_var.get().strip())
        except ValueError as exc:
            raise ValueError("Camera index must be an integer.") from exc

        db_raw = self.db_path_var.get().strip()
        if not db_raw:
            raise ValueError("Database path is required.")
        db_path = Path(db_raw)

        weights_raw = self.weights_path_var.get().strip()
        weights_path = Path(weights_raw) if weights_raw else None

        device = self.device_var.get().strip()
        return camera_index, db_path, weights_path, device

    def _build_base_cmd(self) -> list[str]:
        camera_index, db_path, weights_path, device = self._validate_common()
        cmd = [
            sys.executable,
            "camera_app.py",
            "--camera-index",
            str(camera_index),
            "--db-path",
            str(db_path),
        ]
        if weights_path is not None:
            cmd.extend(["--weights-path", str(weights_path)])
        if device:
            cmd.extend(["--device", device])
        return cmd

    def start_register(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Busy", "A process is already running. Stop it first.")
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Validation Error", "User name is required.")
            return
        try:
            samples = int(self.samples_var.get().strip())
            if samples < 1:
                raise ValueError
            cmd = self._build_base_cmd()
        except ValueError as exc:
            messagebox.showerror("Validation Error", str(exc))
            return
        cmd.extend(["register", "--name", name, "--samples", str(samples)])
        self._start_process(cmd, "REGISTER", "Registration started")

    def start_verify(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Busy", "A process is already running. Stop it first.")
            return
        try:
            threshold = float(self.threshold_var.get().strip())
            event_ttl_days = int(self.event_ttl_days_var.get().strip())
            event_cooldown = float(self.event_cooldown_var.get().strip())
            fast_window = float(self.fast_window_var.get().strip())
            fast_face_jitter = float(self.fast_face_jitter_var.get().strip())
            fast_ear_delta = float(self.fast_ear_delta_var.get().strip())
            passive_residual = float(self.passive_residual_var.get().strip())
            passive_texture = float(self.passive_texture_var.get().strip())
            passive_max_jitter = float(self.passive_max_jitter_var.get().strip())
            telegram_fail_threshold = int(self.telegram_fail_threshold_var.get().strip())
            telegram_window = float(self.telegram_window_var.get().strip())
            telegram_alert_cooldown = float(self.telegram_alert_cooldown_var.get().strip())
            telegram_unknown_cooldown = float(self.telegram_unknown_cooldown_var.get().strip())
            telegram_admin_allow_sec = float(self.telegram_admin_allow_sec_var.get().strip())
            cmd = self._build_base_cmd()
        except ValueError as exc:
            messagebox.showerror("Validation Error", str(exc))
            return

        cmd.extend(["verify", "--threshold", str(threshold)])
        if self.show_score_var.get():
            cmd.append("--show-score")
        cmd.extend(["--event-ttl-days", str(event_ttl_days)])
        cmd.extend(["--event-cooldown-sec", str(event_cooldown)])

        if self.liveness_enabled_var.get():
            cmd.append("--liveness")
            cmd.extend(["--liveness-mode", self.liveness_mode_var.get()])
        else:
            cmd.append("--no-liveness")

        cmd.extend(["--fast-window-sec", str(fast_window)])
        cmd.extend(["--fast-face-jitter-px", str(fast_face_jitter)])
        cmd.extend(["--fast-ear-delta", str(fast_ear_delta)])
        cmd.append("--fast-require-blink" if self.fast_require_blink_var.get() else "--no-fast-require-blink")
        cmd.extend(["--passive-residual-flow", str(passive_residual)])
        cmd.extend(["--passive-min-texture", str(passive_texture)])
        cmd.extend(["--passive-max-face-jitter-px", str(passive_max_jitter)])
        cmd.append("--telegram-enabled" if self.telegram_enabled_var.get() else "--no-telegram-enabled")
        if self.telegram_bot_token_var.get().strip():
            cmd.extend(["--telegram-bot-token", self.telegram_bot_token_var.get().strip()])
        if self.telegram_chat_id_var.get().strip():
            cmd.extend(["--telegram-chat-id", self.telegram_chat_id_var.get().strip()])
        cmd.extend(["--telegram-fail-threshold", str(telegram_fail_threshold)])
        cmd.extend(["--telegram-window-sec", str(telegram_window)])
        cmd.extend(["--telegram-alert-cooldown-sec", str(telegram_alert_cooldown)])
        cmd.extend(["--telegram-unknown-cooldown-sec", str(telegram_unknown_cooldown)])
        cmd.append("--telegram-send-photo" if self.telegram_send_photo_var.get() else "--no-telegram-send-photo")
        cmd.append("--telegram-admin-buttons" if self.telegram_admin_buttons_var.get() else "--no-telegram-admin-buttons")
        cmd.extend(["--telegram-admin-allow-sec", str(telegram_admin_allow_sec)])

        self._start_process(cmd, "VERIFY", "Verification started")

    def download_weights(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Busy", "Stop running process first.")
            return
        cmd = [sys.executable, "download_weights.py"]
        self._start_process(cmd, "SETUP", "Downloading FaceNet weights")

    def test_telegram(self) -> None:
        token = self.telegram_bot_token_var.get().strip()
        chat_id = self.telegram_chat_id_var.get().strip()
        if not token or not chat_id:
            messagebox.showerror(
                "Validation Error",
                "Set Telegram Bot Token and Chat ID first.",
            )
            return
        self.set_status("TELEGRAM", "Sending test message...")
        self.append_log(">>> test telegram notification")
        threading.Thread(
            target=self._send_test_telegram,
            args=(token, chat_id),
            daemon=True,
        ).start()

    def _send_test_telegram(self, token: str, chat_id: str) -> None:
        try:
            TelegramAlerter(bot_token=token, chat_id=chat_id).send(
                "FaceID test alert: Telegram integration is active."
            )
        except Exception as exc:
            self.after(0, self.append_log, f"telegram test failed: {exc}")
            self.after(0, self.set_status, "ERROR", "Telegram test failed")
            return
        self.after(0, self.append_log, "telegram test sent successfully")
        self.after(0, self.set_status, "TELEGRAM", "Telegram test sent")

    def _start_process(self, cmd: list[str], mode: str, status: str) -> None:
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(Path(__file__).parent),
            )
        except Exception as exc:
            self.proc = None
            messagebox.showerror("Process Start Error", str(exc))
            return

        self.append_log(">>> " + " ".join(cmd))
        self.set_status(mode, status)
        self.output_thread = threading.Thread(target=self._read_output, daemon=True)
        self.output_thread.start()

    def _read_output(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            text = line.rstrip("\n")
            self.after(0, self.append_log, text)
            self.after(0, self._update_status_from_log, text)
        exit_code = proc.wait()
        self.after(0, self._on_process_exit, exit_code)

    def _update_status_from_log(self, text: str) -> None:
        if "ALLOW" in text:
            self.set_status("VERIFY", "Access granted")
        elif "DENY_" in text:
            self.set_status("VERIFY", "Access denied")
        elif "error" in text.lower() or "traceback" in text.lower():
            self.set_status("ERROR", "Check log for details")

    def _on_process_exit(self, exit_code: int) -> None:
        self.append_log(f"[process exited with code {exit_code}]")
        self.proc = None
        if exit_code == 0:
            self.set_status("IDLE", "Ready")
        else:
            self.set_status("ERROR", f"Process failed (code {exit_code})")

    def stop_process(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        self.append_log("[terminate requested]")
        self.set_status("STOPPING", "Stopping process...")

    def _on_close(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
        self.destroy()

    def _refresh_events(self) -> None:
        db_path = self.db_path_var.get().strip()
        rows: list[tuple[str, str, str, float]] = []
        if db_path:
            try:
                conn = sqlite3.connect(db_path)
                try:
                    rows = conn.execute(
                        """
                        SELECT ts_utc, decision, identity, score
                        FROM access_events
                        ORDER BY id DESC
                        LIMIT 30
                        """
                    ).fetchall()
                finally:
                    conn.close()
            except Exception:
                rows = []

        self.events_tree.delete(*self.events_tree.get_children())
        for ts_utc, decision, identity, score in rows:
            time_short = str(ts_utc)[11:19] if len(str(ts_utc)) >= 19 else str(ts_utc)
            score_text = "-" if score is None or float(score) < 0 else f"{float(score):.3f}"
            self.events_tree.insert(
                "",
                "end",
                values=(time_short, decision, identity, score_text),
            )

        self.after(self.events_refresh_ms, self._refresh_events)


def main() -> None:
    app = FaceIdGui()
    app.mainloop()


if __name__ == "__main__":
    main()
