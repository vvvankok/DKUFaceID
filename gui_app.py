from __future__ import annotations

import datetime
import sqlite3
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from src.faceid.alerts import TelegramAlerter


# ─────────────────────────── colour palette ───────────────────────────────────
C_BG        = "#f0f4f8"
C_HEADER    = "#1e293b"
C_ACCENT    = "#4f46e5"
C_ACCENT2   = "#6366f1"
C_SUCCESS   = "#16a34a"
C_ERROR     = "#dc2626"
C_WARN      = "#d97706"
C_WHITE     = "#ffffff"
C_SURFACE   = "#f8fafc"
C_BORDER    = "#e2e8f0"
C_TEXT      = "#0f172a"
C_MUTED     = "#64748b"
C_LOG_BG    = "#0f172a"
C_LOG_FG    = "#e2e8f0"

# row tags for events table
ROW_ALLOW   = {"bg": "#dcfce7", "fg": "#14532d"}
ROW_DENY    = {"bg": "#fee2e2", "fg": "#7f1d1d"}
ROW_LIVE    = {"bg": "#ffedd5", "fg": "#7c2d12"}
ROW_OTHER   = {"bg": "#f1f5f9", "fg": "#334155"}


class FaceIdGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FaceID — Система контроля доступа")
        self.geometry("1100x750")
        self.minsize(960, 640)
        self.configure(bg=C_BG)

        self.proc: subprocess.Popen[str] | None = None
        self.output_thread: threading.Thread | None = None

        self._init_vars()
        self._load_config()
        self._setup_style()
        self._build_ui()
        self._refresh_events()
        self._refresh_db_tab()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── variables ─────────────────────────────────────────────────────────────
    def _init_vars(self) -> None:
        self.name_var                    = tk.StringVar(value="")
        self.samples_var                 = tk.StringVar(value="20")
        self.threshold_var               = tk.StringVar(value="0.72")
        self.db_path_var                 = tk.StringVar(value="artifacts/faceid_identities.db")
        self.weights_path_var            = tk.StringVar(value="")
        self.anti_spoof_model_path_var   = tk.StringVar(value="artifacts/anti_spoof_model.pt")
        self.camera_index_var            = tk.StringVar(value="0")
        self.device_var                  = tk.StringVar(value="")

        self.liveness_mode_var           = tk.StringVar(value="passive")
        self.liveness_enabled_var        = tk.BooleanVar(value=True)
        self.show_score_var              = tk.BooleanVar(value=True)
        self.event_ttl_days_var          = tk.StringVar(value="7")
        self.event_cooldown_var          = tk.StringVar(value="1.2")

        self.fast_window_var             = tk.StringVar(value="0.8")
        self.fast_face_jitter_var        = tk.StringVar(value="6")
        self.fast_ear_delta_var          = tk.StringVar(value="0.018")
        self.fast_require_blink_var      = tk.BooleanVar(value=True)
        self.passive_residual_var        = tk.StringVar(value="0.24")
        self.passive_texture_var         = tk.StringVar(value="45")
        self.passive_max_jitter_var      = tk.StringVar(value="8")
        self.anti_spoof_enabled_var      = tk.BooleanVar(value=True)
        self.anti_spoof_threshold_var    = tk.StringVar(value="0.80")

        self.relay_enabled_var           = tk.BooleanVar(value=False)
        self.relay_pin_var               = tk.StringVar(value="18")
        self.relay_duration_var          = tk.StringVar(value="2.0")
        self.relay_active_high_var       = tk.BooleanVar(value=True)

        self.telegram_enabled_var        = tk.BooleanVar(value=False)
        self.telegram_bot_token_var      = tk.StringVar(value="")
        self.telegram_chat_id_var        = tk.StringVar(value="")
        self.telegram_chat_id2_var       = tk.StringVar(value="")
        self.telegram_fail_threshold_var = tk.StringVar(value="3")
        self.telegram_window_var         = tk.StringVar(value="20")
        self.telegram_alert_cooldown_var = tk.StringVar(value="40")
        self.telegram_unknown_cooldown_var = tk.StringVar(value="20")
        self.telegram_send_photo_var     = tk.BooleanVar(value=True)
        self.telegram_admin_buttons_var  = tk.BooleanVar(value=True)
        self.telegram_admin_allow_sec_var = tk.StringVar(value="15")

        self.key_path_var                = tk.StringVar(value="artifacts/faceid.key")

        self.status_var                  = tk.StringVar(value="Ожидание")
        self.mode_var                    = tk.StringVar(value="IDLE")
        self.events_refresh_ms           = 1500

    # ── ttk style ─────────────────────────────────────────────────────────────
    def _setup_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure(".",
            background=C_BG, foreground=C_TEXT,
            font=("Segoe UI", 10),
        )
        style.configure("TFrame", background=C_BG)
        style.configure("TLabel", background=C_BG, foreground=C_TEXT)
        style.configure("TCheckbutton", background=C_BG, foreground=C_TEXT)
        style.configure("TLabelframe",
            background=C_BG, foreground=C_TEXT,
            bordercolor=C_BORDER, relief="flat",
        )
        style.configure("TLabelframe.Label",
            background=C_BG, foreground=C_ACCENT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("TCombobox", fieldbackground=C_WHITE, background=C_WHITE)
        style.configure("TEntry", fieldbackground=C_WHITE)
        style.configure("TScrollbar", background=C_BG, troughcolor=C_BORDER)

        # Notebook tabs
        style.configure("TNotebook", background=C_BG, tabmargins=[2, 5, 2, 0])
        style.configure("TNotebook.Tab",
            background=C_BORDER, foreground=C_MUTED,
            padding=[16, 8], font=("Segoe UI", 10),
        )
        style.map("TNotebook.Tab",
            background=[("selected", C_WHITE)],
            foreground=[("selected", C_ACCENT)],
            font=[("selected", ("Segoe UI", 10, "bold"))],
        )

        # Buttons
        style.configure("TButton",
            background=C_ACCENT, foreground=C_WHITE,
            relief="flat", padding=[14, 6],
            font=("Segoe UI", 10, "bold"),
        )
        style.map("TButton",
            background=[("active", C_ACCENT2), ("disabled", C_BORDER)],
            foreground=[("disabled", C_MUTED)],
        )
        style.configure("Danger.TButton",
            background=C_ERROR, foreground=C_WHITE,
        )
        style.map("Danger.TButton",
            background=[("active", "#b91c1c")],
        )
        style.configure("Success.TButton",
            background=C_SUCCESS, foreground=C_WHITE,
        )
        style.map("Success.TButton",
            background=[("active", "#15803d")],
        )
        style.configure("Neutral.TButton",
            background="#94a3b8", foreground=C_WHITE,
        )
        style.map("Neutral.TButton",
            background=[("active", "#64748b")],
        )

        # Status bar pill
        style.configure("Status.TLabel",
            background=C_HEADER, foreground=C_WHITE,
            font=("Segoe UI", 10, "bold"), padding=[14, 5],
        )
        style.configure("StatusText.TLabel",
            background=C_HEADER, foreground="#94a3b8",
            font=("Segoe UI", 10), padding=[8, 5],
        )
        style.configure("Card.TFrame",
            background=C_WHITE, relief="flat",
        )
        style.configure("Muted.TLabel",
            foreground=C_MUTED, font=("Segoe UI", 9),
        )

        # Treeview
        style.configure("Treeview",
            background=C_WHITE, fieldbackground=C_WHITE,
            foreground=C_TEXT, font=("Segoe UI", 10),
            rowheight=26,
        )
        style.configure("Treeview.Heading",
            background=C_HEADER, foreground=C_WHITE,
            font=("Segoe UI", 10, "bold"), relief="flat",
        )
        style.map("Treeview",
            background=[("selected", C_ACCENT2)],
            foreground=[("selected", C_WHITE)],
        )

    # ── top-level layout ───────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        # ── header bar ────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=C_HEADER, height=58)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        tk.Label(
            header, text="  FaceID",
            bg=C_HEADER, fg=C_WHITE,
            font=("Segoe UI", 18, "bold"),
        ).pack(side=tk.LEFT, padx=(16, 4), pady=10)
        tk.Label(
            header, text="Система контроля доступа",
            bg=C_HEADER, fg="#94a3b8",
            font=("Segoe UI", 11),
        ).pack(side=tk.LEFT, padx=(0, 0), pady=10)

        # status pill on the right
        self._status_pill = tk.Label(
            header, textvariable=self.mode_var,
            bg="#374151", fg="#f9fafb",
            font=("Segoe UI", 10, "bold"),
            padx=14, pady=4,
        )
        self._status_pill.pack(side=tk.RIGHT, padx=16, pady=12)
        self._status_text_lbl = tk.Label(
            header, textvariable=self.status_var,
            bg=C_HEADER, fg="#94a3b8",
            font=("Segoe UI", 10),
        )
        self._status_text_lbl.pack(side=tk.RIGHT, padx=(0, 4), pady=12)

        # ── stop button always visible ─────────────────────────────────────
        ttk.Button(
            header, text="Стоп", style="Danger.TButton",
            command=self.stop_process,
        ).pack(side=tk.RIGHT, padx=(0, 8), pady=12)

        # ── notebook ──────────────────────────────────────────────────────────
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 0))

        self._tab_main     = ttk.Frame(self.nb, padding=10)
        self._tab_reg      = ttk.Frame(self.nb, padding=10)
        self._tab_verify   = ttk.Frame(self.nb, padding=10)
        self._tab_telegram = ttk.Frame(self.nb, padding=10)
        self._tab_db       = ttk.Frame(self.nb, padding=10)

        self.nb.add(self._tab_main,     text="  Главная  ")
        self.nb.add(self._tab_reg,      text="  Регистрация  ")
        self.nb.add(self._tab_verify,   text="  Верификация  ")
        self.nb.add(self._tab_telegram, text="  Telegram  ")
        self.nb.add(self._tab_db,       text="  База данных  ")

        self._build_tab_main()
        self._build_tab_reg()
        self._build_tab_verify()
        self._build_tab_telegram()
        self._build_tab_db()

    # ══════════════════════ TAB: Главная ═════════════════════════════════════
    def _build_tab_main(self) -> None:
        t = self._tab_main
        t.columnconfigure(0, weight=3)
        t.columnconfigure(1, weight=2)
        t.rowconfigure(0, weight=0)
        t.rowconfigure(1, weight=1)

        # ── quick settings row ────────────────────────────────────────────────
        qs = ttk.LabelFrame(t, text="Быстрые настройки", padding=10)
        qs.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        qs.columnconfigure(1, weight=1)
        qs.columnconfigure(3, weight=1)
        qs.columnconfigure(5, weight=1)

        ttk.Label(qs, text="Камера").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(qs, textvariable=self.camera_index_var, width=6).grid(row=0, column=1, sticky="ew", padx=(0, 16))
        ttk.Label(qs, text="База данных").grid(row=0, column=2, sticky="w", padx=(0, 6))
        ttk.Entry(qs, textvariable=self.db_path_var).grid(row=0, column=3, sticky="ew", padx=(0, 16))
        ttk.Label(qs, text="Устройство").grid(row=0, column=4, sticky="w", padx=(0, 6))
        ttk.Entry(qs, textvariable=self.device_var, width=8).grid(row=0, column=5, sticky="ew")

        # ── action buttons ────────────────────────────────────────────────────
        btn_row = ttk.Frame(t)
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Button(
            btn_row, text="▶  Регистрация", style="TButton",
            command=self.start_register,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            btn_row, text="▶  Верификация", style="Success.TButton",
            command=self.start_verify,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            btn_row, text="Скачать веса", style="Neutral.TButton",
            command=self.download_weights,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            btn_row, text="Тест Telegram", style="Neutral.TButton",
            command=self.test_telegram,
        ).pack(side=tk.LEFT)

        # ── log ───────────────────────────────────────────────────────────────
        log_card = ttk.LabelFrame(t, text="Журнал", padding=6)
        log_card.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        log_card.rowconfigure(0, weight=1)
        log_card.columnconfigure(0, weight=1)
        t.rowconfigure(2, weight=1)

        self.log = tk.Text(
            log_card, wrap=tk.WORD,
            bg=C_LOG_BG, fg=C_LOG_FG,
            insertbackground=C_LOG_FG, relief=tk.FLAT,
            font=("Consolas", 10), state=tk.DISABLED,
        )
        log_scroll = ttk.Scrollbar(log_card, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

        # ── events table ──────────────────────────────────────────────────────
        ev_card = ttk.LabelFrame(t, text="Последние события", padding=6)
        ev_card.grid(row=2, column=1, sticky="nsew")
        ev_card.rowconfigure(0, weight=1)
        ev_card.columnconfigure(0, weight=1)

        cols = ("time", "decision", "identity", "score")
        self.events_tree = ttk.Treeview(ev_card, columns=cols, show="headings", height=16)
        self.events_tree.heading("time",     text="Время")
        self.events_tree.heading("decision", text="Решение")
        self.events_tree.heading("identity", text="Пользователь")
        self.events_tree.heading("score",    text="Схожесть")
        self.events_tree.column("time",     width=80,  anchor="w")
        self.events_tree.column("decision", width=110, anchor="w")
        self.events_tree.column("identity", width=100, anchor="w")
        self.events_tree.column("score",    width=70,  anchor="e")

        # colour tags
        self.events_tree.tag_configure("allow", background=ROW_ALLOW["bg"], foreground=ROW_ALLOW["fg"])
        self.events_tree.tag_configure("deny",  background=ROW_DENY["bg"],  foreground=ROW_DENY["fg"])
        self.events_tree.tag_configure("live",  background=ROW_LIVE["bg"],  foreground=ROW_LIVE["fg"])
        self.events_tree.tag_configure("other", background=ROW_OTHER["bg"], foreground=ROW_OTHER["fg"])

        ev_scroll = ttk.Scrollbar(ev_card, orient=tk.VERTICAL, command=self.events_tree.yview)
        self.events_tree.configure(yscrollcommand=ev_scroll.set)
        self.events_tree.grid(row=0, column=0, sticky="nsew")
        ev_scroll.grid(row=0, column=1, sticky="ns")

    # ══════════════════════ TAB: Регистрация ══════════════════════════════════
    def _build_tab_reg(self) -> None:
        t = self._tab_reg
        t.columnconfigure(0, weight=1)
        t.columnconfigure(1, weight=1)

        # ── common camera settings ────────────────────────────────────────────
        env = ttk.LabelFrame(t, text="Окружение", padding=12)
        env.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        env.columnconfigure(1, weight=1)

        self._field(env, 0, "Индекс камеры",          self.camera_index_var)
        self._field(env, 1, "База данных (.db)",       self.db_path_var)
        self._field(env, 2, "Веса FaceNet (.pt)",      self.weights_path_var, browse=True)
        self._field(env, 3, "Устройство (cpu/cuda)",   self.device_var)

        # ── registration params ───────────────────────────────────────────────
        reg = ttk.LabelFrame(t, text="Параметры регистрации", padding=12)
        reg.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        reg.columnconfigure(1, weight=1)

        self._field(reg, 0, "Имя пользователя", self.name_var)
        self._field(reg, 1, "Кол-во снимков",   self.samples_var)
        ttk.Label(
            reg, text="Рекомендуется 20–30 снимков для стабильного распознавания.",
            style="Muted.TLabel", wraplength=280,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Button(
            reg, text="▶  Начать регистрацию",
            command=self.start_register,
        ).grid(row=3, column=0, columnspan=2, pady=(14, 0), sticky="ew")

        # ── tips ──────────────────────────────────────────────────────────────
        tips = ttk.LabelFrame(t, text="Советы", padding=12)
        tips.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        ttk.Label(
            tips,
            text=(
                "Для лучшего результата:\n\n"
                "• Хорошее, равномерное освещение\n"
                "• Разные углы лица при регистрации\n"
                "• Расстояние 40–70 см до камеры\n"
                "• Избегайте блеска на очках\n"
                "• Не закрывайте лицо аксессуарами"
            ),
            style="Muted.TLabel", justify=tk.LEFT,
        ).pack(anchor="nw")

        t.rowconfigure(1, weight=1)

    # ══════════════════════ TAB: Верификация ══════════════════════════════════
    def _build_tab_verify(self) -> None:
        t = self._tab_verify

        canvas = tk.Canvas(t, bg=C_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(t, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win_id, width=e.width))
        self.bind_all("<MouseWheel>",  lambda e: self._scroll_canvas_event(canvas, e))

        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

        # ── main verify params ────────────────────────────────────────────────
        vp = ttk.LabelFrame(inner, text="Основные параметры", padding=12)
        vp.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        vp.columnconfigure(1, weight=1)

        self._field(vp, 0, "Порог схожести (0–1)", self.threshold_var)
        self._field(vp, 1, "Антиспуф-модель (.pt)", self.anti_spoof_model_path_var, browse=True)
        self._field(vp, 2, "TTL событий (дней)",    self.event_ttl_days_var)
        self._field(vp, 3, "Кулдаун событий (сек)", self.event_cooldown_var)

        ttk.Label(vp, text="Режим живости").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Combobox(
            vp, textvariable=self.liveness_mode_var,
            values=["passive", "fast", "blink", "motion"], state="readonly",
        ).grid(row=4, column=1, sticky="ew", pady=4)

        chk_row = ttk.Frame(vp)
        chk_row.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(chk_row, text="Живость",         variable=self.liveness_enabled_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(chk_row, text="Показывать балл", variable=self.show_score_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(chk_row, text="Антиспуф-модель", variable=self.anti_spoof_enabled_var).pack(side=tk.LEFT)

        self._field(vp, 6, "Порог антиспуфа (0–1)", self.anti_spoof_threshold_var)

        # ── fast liveness ─────────────────────────────────────────────────────
        fl = ttk.LabelFrame(inner, text="Быстрая живость (fast)", padding=10)
        fl.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 10))
        fl.columnconfigure(1, weight=1)
        self._field(fl, 0, "Окно (сек)",           self.fast_window_var)
        self._field(fl, 1, "Движение лица (px)",   self.fast_face_jitter_var)
        self._field(fl, 2, "EAR дельта",           self.fast_ear_delta_var)
        ttk.Checkbutton(fl, text="Требовать моргание", variable=self.fast_require_blink_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # ── passive liveness ──────────────────────────────────────────────────
        pl = ttk.LabelFrame(inner, text="Пассивная живость (passive)", padding=10)
        pl.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(0, 10))
        pl.columnconfigure(1, weight=1)
        self._field(pl, 0, "Остаточный поток",   self.passive_residual_var)
        self._field(pl, 1, "Мин. текстура",      self.passive_texture_var)
        self._field(pl, 2, "Макс. джиттер (px)", self.passive_max_jitter_var)

        # ── relay ─────────────────────────────────────────────────────────────
        rl = ttk.LabelFrame(inner, text="Управление реле / замком (ТЗ п. 4.1)", padding=10)
        rl.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        rl.columnconfigure(1, weight=1)

        ttk.Checkbutton(rl, text="Включить управление реле", variable=self.relay_enabled_var).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self._field(rl, 1, "GPIO пин (BCM)",          self.relay_pin_var)
        self._field(rl, 2, "Время открытия (сек)",    self.relay_duration_var)
        ttk.Checkbutton(rl, text="Активный HIGH (по умолчанию)", variable=self.relay_active_high_var).grid(
            row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(
            rl,
            text="На Raspberry Pi управляет GPIO-реле. На ПК — симуляция в журнале.",
            style="Muted.TLabel",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Button(
            inner, text="▶  Начать верификацию", style="Success.TButton",
            command=self.start_verify,
        ).grid(row=3, column=0, columnspan=2, pady=(6, 0), sticky="ew")

    # ══════════════════════ TAB: Telegram ═════════════════════════════════════
    def _build_tab_telegram(self) -> None:
        t = self._tab_telegram
        t.columnconfigure(0, weight=1)
        t.columnconfigure(1, weight=1)

        # ── connection ────────────────────────────────────────────────────────
        conn = ttk.LabelFrame(t, text="Подключение", padding=12)
        conn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        conn.columnconfigure(1, weight=1)

        ttk.Checkbutton(conn, text="Включить уведомления Telegram", variable=self.telegram_enabled_var).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(conn, text="Bot Token").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        token_entry = ttk.Entry(conn, textvariable=self.telegram_bot_token_var, show="●")
        token_entry.grid(row=1, column=1, sticky="ew", pady=4)
        # toggle visibility
        self._token_visible = False
        def toggle_token():
            self._token_visible = not self._token_visible
            token_entry.config(show="" if self._token_visible else "●")
            btn_show.config(text="Скрыть" if self._token_visible else "Показать")
        btn_show = ttk.Button(conn, text="Показать", style="Neutral.TButton", command=toggle_token)
        btn_show.grid(row=1, column=2, padx=(8, 0), pady=4)

        self._field(conn, 2, "Chat ID 1", self.telegram_chat_id_var)
        self._field(conn, 3, "Chat ID 2 (необяз.)", self.telegram_chat_id2_var)

        # ── alert settings ────────────────────────────────────────────────────
        al = ttk.LabelFrame(t, text="Параметры уведомлений", padding=12)
        al.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        al.columnconfigure(1, weight=1)

        self._field(al, 0, "Порог отказов",          self.telegram_fail_threshold_var)
        self._field(al, 1, "Окно (сек)",              self.telegram_window_var)
        self._field(al, 2, "Кулдаун алертов (сек)",  self.telegram_alert_cooldown_var)
        self._field(al, 3, "Кулдаун unknown (сек)",  self.telegram_unknown_cooldown_var)
        self._field(al, 4, "Разрешение admin (сек)", self.telegram_admin_allow_sec_var)

        # ── options ───────────────────────────────────────────────────────────
        op = ttk.LabelFrame(t, text="Опции", padding=12)
        op.grid(row=1, column=1, sticky="nsew", padx=(8, 0), pady=(0, 10))

        ttk.Checkbutton(op, text="Отправлять фото с алертом",      variable=self.telegram_send_photo_var).pack(anchor="w", pady=3)
        ttk.Checkbutton(op, text="Кнопки Разрешить / Отклонить", variable=self.telegram_admin_buttons_var).pack(anchor="w", pady=3)

        ttk.Label(
            op,
            text=(
                "Доступные команды бота:\n\n"
                "/help — справка\n"
                "/list — список пользователей\n"
                "/stats — статистика за сегодня\n"
                "/attendance — посещаемость\n"
                "/status — статус системы"
            ),
            style="Muted.TLabel", justify=tk.LEFT,
        ).pack(anchor="nw", pady=(12, 0))

        # test button
        ttk.Button(
            t, text="Отправить тестовое сообщение", style="Neutral.TButton",
            command=self.test_telegram,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 0))

        t.rowconfigure(1, weight=1)

    # ══════════════════════ TAB: База данных ══════════════════════════════════
    def _build_tab_db(self) -> None:
        t = self._tab_db
        t.columnconfigure(0, weight=1)
        t.columnconfigure(1, weight=1)
        t.columnconfigure(2, weight=1)
        t.rowconfigure(1, weight=1)

        # ── header buttons ────────────────────────────────────────────────────
        btn_row = ttk.Frame(t)
        btn_row.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        ttk.Button(btn_row, text="Обновить", style="Neutral.TButton",
                   command=self._refresh_db_tab).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Удалить выбранного", style="Danger.TButton",
                   command=self._delete_identity).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Экспорт CSV", style="Neutral.TButton",
                   command=self._export_csv).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Резервная копия", style="Neutral.TButton",
                   command=self._backup_db).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Зашифр. копия", style="Neutral.TButton",
                   command=self._backup_db_encrypted).pack(side=tk.LEFT)

        # ── registered users ──────────────────────────────────────────────────
        users_card = ttk.LabelFrame(t, text="Зарегистрированные пользователи", padding=8)
        users_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        users_card.rowconfigure(0, weight=1)
        users_card.columnconfigure(0, weight=1)

        ucols = ("name", "updated")
        self.users_tree = ttk.Treeview(users_card, columns=ucols, show="headings")
        self.users_tree.heading("name",    text="Имя")
        self.users_tree.heading("updated", text="Обновлено")
        self.users_tree.column("name",    width=140, anchor="w")
        self.users_tree.column("updated", width=110, anchor="w")
        u_scroll = ttk.Scrollbar(users_card, orient=tk.VERTICAL, command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=u_scroll.set)
        self.users_tree.grid(row=0, column=0, sticky="nsew")
        u_scroll.grid(row=0, column=1, sticky="ns")

        # ── today attendance ──────────────────────────────────────────────────
        attend_card = ttk.LabelFrame(t, text="Посещаемость сегодня", padding=8)
        attend_card.grid(row=1, column=1, sticky="nsew", padx=(0, 6))
        attend_card.rowconfigure(0, weight=1)
        attend_card.columnconfigure(0, weight=1)

        acols = ("name", "arrived")
        self.attend_tree = ttk.Treeview(attend_card, columns=acols, show="headings")
        self.attend_tree.heading("name",    text="Имя")
        self.attend_tree.heading("arrived", text="Вошёл (UTC)")
        self.attend_tree.column("name",    width=130, anchor="w")
        self.attend_tree.column("arrived", width=90,  anchor="center")
        a_scroll = ttk.Scrollbar(attend_card, orient=tk.VERTICAL, command=self.attend_tree.yview)
        self.attend_tree.configure(yscrollcommand=a_scroll.set)
        self.attend_tree.grid(row=0, column=0, sticky="nsew")
        a_scroll.grid(row=0, column=1, sticky="ns")

        # ── today stats ───────────────────────────────────────────────────────
        stats_card = ttk.LabelFrame(t, text="Статистика за сегодня", padding=12)
        stats_card.grid(row=1, column=2, sticky="nsew", padx=(0, 0))
        stats_card.columnconfigure(0, weight=1)

        self.stats_allow_var    = tk.StringVar(value="—")
        self.stats_deny_var     = tk.StringVar(value="—")
        self.stats_sus_var      = tk.StringVar(value="—")
        self.stats_total_var    = tk.StringVar(value="—")
        self.stats_visitors_var = tk.StringVar(value="—")

        def stat_row(parent: ttk.Frame, row: int, label: str, var: tk.StringVar, color: str) -> None:
            f = ttk.Frame(parent)
            f.grid(row=row, column=0, sticky="ew", pady=4)
            f.columnconfigure(1, weight=1)
            tk.Label(f, text=label, bg=C_BG, fg=C_MUTED, font=("Segoe UI", 10)).pack(side=tk.LEFT)
            tk.Label(f, textvariable=var, bg=C_BG, fg=color, font=("Segoe UI", 18, "bold")).pack(side=tk.RIGHT)

        stat_row(stats_card, 0, "👤  Посетителей:",      self.stats_visitors_var, C_ACCENT)
        stat_row(stats_card, 1, "✅  Разрешено:",         self.stats_allow_var,    C_SUCCESS)
        stat_row(stats_card, 2, "❌  Отказов:",           self.stats_deny_var,     C_ERROR)
        stat_row(stats_card, 3, "⚠️  Подозрительных:",   self.stats_sus_var,      C_WARN)
        stat_row(stats_card, 4, "📊  Всего событий:",    self.stats_total_var,    C_MUTED)

        ttk.Button(
            stats_card, text="Обновить", style="Neutral.TButton",
            command=self._refresh_db_tab,
        ).grid(row=5, column=0, sticky="ew", pady=(12, 0))

    # ── helper: field row ──────────────────────────────────────────────────────
    def _field(
        self, parent: ttk.Frame, row: int, label: str,
        var: tk.Variable, browse: bool = False,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        if browse:
            ttk.Button(
                parent, text="Обзор...", style="Neutral.TButton",
                command=lambda: self._pick_file(var),
            ).grid(row=row, column=2, sticky="ew", padx=(8, 0), pady=4)

    def _pick_file(self, target_var: tk.Variable) -> None:
        path = filedialog.askopenfilename(
            title="Выберите файл весов",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
        )
        if path:
            target_var.set(path)

    def _scroll_canvas_event(self, canvas: tk.Canvas, event: tk.Event) -> None:
        widget = event.widget
        if widget is not None and str(widget.winfo_class()) in {
            "Text", "Treeview", "Entry", "TEntry", "Combobox", "TCombobox"
        }:
            return
        delta = int(-event.delta / 120) if event.delta else 0
        if delta:
            canvas.yview_scroll(delta, "units")

    # ── log / status ───────────────────────────────────────────────────────────
    def append_log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.log.insert(tk.END, f"[{ts}] {text}\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def set_status(self, mode: str, text: str) -> None:
        self.mode_var.set(f"  {mode}  ")
        self.status_var.set(text)
        colors = {
            "REGISTER": ("#7c3aed", C_WHITE),
            "VERIFY":   (C_SUCCESS, C_WHITE),
            "IDLE":     ("#374151", "#f9fafb"),
            "ERROR":    (C_ERROR, C_WHITE),
            "STOPPING": (C_WARN, C_WHITE),
            "SETUP":    (C_ACCENT, C_WHITE),
            "TELEGRAM": ("#0ea5e9", C_WHITE),
        }
        bg, fg = colors.get(mode, ("#374151", "#f9fafb"))
        self._status_pill.configure(bg=bg, fg=fg)

    # ── validation helpers ─────────────────────────────────────────────────────
    def _validate_common(self) -> tuple[int, Path, Path | None, str]:
        try:
            camera_index = int(self.camera_index_var.get().strip())
        except ValueError as exc:
            raise ValueError("Индекс камеры должен быть числом.") from exc
        db_raw = self.db_path_var.get().strip()
        if not db_raw:
            raise ValueError("Путь к базе данных обязателен.")
        weights_raw = self.weights_path_var.get().strip()
        weights_path = Path(weights_raw) if weights_raw else None
        device = self.device_var.get().strip()
        return camera_index, Path(db_raw), weights_path, device

    def _build_base_cmd(self) -> list[str]:
        camera_index, db_path, weights_path, device = self._validate_common()
        cmd = [
            sys.executable, "camera_app.py",
            "--camera-index", str(camera_index),
            "--db-path", str(db_path),
        ]
        if weights_path:
            cmd.extend(["--weights-path", str(weights_path)])
        if device:
            cmd.extend(["--device", device])
        return cmd

    # ── actions ────────────────────────────────────────────────────────────────
    def start_register(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Занято", "Процесс уже запущен. Остановите его.")
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Ошибка", "Введите имя пользователя.")
            return
        try:
            samples = int(self.samples_var.get().strip())
            if samples < 1:
                raise ValueError
            cmd = self._build_base_cmd()
        except ValueError as exc:
            messagebox.showerror("Ошибка", str(exc) or "Некорректные параметры.")
            return
        cmd.extend(["register", "--name", name, "--samples", str(samples)])
        self._start_process(cmd, "REGISTER", f"Регистрация: {name}")
        self.nb.select(0)

    def start_verify(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Занято", "Процесс уже запущен. Остановите его.")
            return
        try:
            threshold            = float(self.threshold_var.get().strip())
            event_ttl_days       = int(self.event_ttl_days_var.get().strip())
            event_cooldown       = float(self.event_cooldown_var.get().strip())
            fast_window          = float(self.fast_window_var.get().strip())
            fast_face_jitter     = float(self.fast_face_jitter_var.get().strip())
            fast_ear_delta       = float(self.fast_ear_delta_var.get().strip())
            passive_residual     = float(self.passive_residual_var.get().strip())
            passive_texture      = float(self.passive_texture_var.get().strip())
            passive_max_jitter   = float(self.passive_max_jitter_var.get().strip())
            anti_spoof_threshold = float(self.anti_spoof_threshold_var.get().strip())
            relay_pin            = int(self.relay_pin_var.get().strip())
            relay_duration       = float(self.relay_duration_var.get().strip())
            tg_fail_threshold    = int(self.telegram_fail_threshold_var.get().strip())
            tg_window            = float(self.telegram_window_var.get().strip())
            tg_alert_cooldown    = float(self.telegram_alert_cooldown_var.get().strip())
            tg_unknown_cooldown  = float(self.telegram_unknown_cooldown_var.get().strip())
            tg_admin_allow_sec   = float(self.telegram_admin_allow_sec_var.get().strip())
            cmd = self._build_base_cmd()
        except ValueError as exc:
            messagebox.showerror("Ошибка", str(exc) or "Некорректные параметры.")
            return

        cmd.extend(["verify", "--threshold", str(threshold)])

        anti_spoof_model_raw = self.anti_spoof_model_path_var.get().strip()
        anti_spoof_enabled   = self.anti_spoof_enabled_var.get()
        if anti_spoof_enabled and anti_spoof_model_raw and not Path(anti_spoof_model_raw).exists():
            messagebox.showwarning(
                "Антиспуф-модель не найдена",
                "Файл модели не существует.\nВерификация запустится без антиспуфа.",
            )
            anti_spoof_enabled = False
        elif anti_spoof_enabled and not anti_spoof_model_raw:
            anti_spoof_enabled = False

        if anti_spoof_model_raw:
            cmd.extend(["--anti-spoof-model-path", anti_spoof_model_raw])
        if self.show_score_var.get():
            cmd.append("--show-score")
        cmd.extend(["--event-ttl-days", str(event_ttl_days)])
        cmd.extend(["--event-cooldown-sec", str(event_cooldown)])
        cmd.append("--liveness" if self.liveness_enabled_var.get() else "--no-liveness")
        cmd.extend(["--liveness-mode", self.liveness_mode_var.get()])
        cmd.append("--anti-spoof" if anti_spoof_enabled else "--no-anti-spoof")
        cmd.extend(["--anti-spoof-threshold", str(anti_spoof_threshold)])
        cmd.extend(["--fast-window-sec", str(fast_window)])
        cmd.extend(["--fast-face-jitter-px", str(fast_face_jitter)])
        cmd.extend(["--fast-ear-delta", str(fast_ear_delta)])
        cmd.append("--fast-require-blink" if self.fast_require_blink_var.get() else "--no-fast-require-blink")
        cmd.extend(["--passive-residual-flow", str(passive_residual)])
        cmd.extend(["--passive-min-texture", str(passive_texture)])
        cmd.extend(["--passive-max-face-jitter-px", str(passive_max_jitter)])

        # relay args
        cmd.append("--relay-enabled" if self.relay_enabled_var.get() else "--no-relay-enabled")
        cmd.extend(["--relay-pin", str(relay_pin)])
        cmd.extend(["--relay-duration-sec", str(relay_duration)])
        cmd.append("--relay-active-high" if self.relay_active_high_var.get() else "--no-relay-active-high")

        # telegram args
        cmd.append("--telegram-enabled" if self.telegram_enabled_var.get() else "--no-telegram-enabled")
        if self.telegram_bot_token_var.get().strip():
            cmd.extend(["--telegram-bot-token", self.telegram_bot_token_var.get().strip()])
        if self.telegram_chat_id_var.get().strip():
            cmd.extend(["--telegram-chat-id", self.telegram_chat_id_var.get().strip()])
        if self.telegram_chat_id2_var.get().strip():
            cmd.extend(["--telegram-chat-id2", self.telegram_chat_id2_var.get().strip()])
        cmd.extend(["--telegram-fail-threshold",   str(tg_fail_threshold)])
        cmd.extend(["--telegram-window-sec",        str(tg_window)])
        cmd.extend(["--telegram-alert-cooldown-sec",str(tg_alert_cooldown)])
        cmd.extend(["--telegram-unknown-cooldown-sec", str(tg_unknown_cooldown)])
        cmd.append("--telegram-send-photo" if self.telegram_send_photo_var.get() else "--no-telegram-send-photo")
        cmd.append("--telegram-admin-buttons" if self.telegram_admin_buttons_var.get() else "--no-telegram-admin-buttons")
        cmd.extend(["--telegram-admin-allow-sec", str(tg_admin_allow_sec)])

        self._start_process(cmd, "VERIFY", "Верификация запущена")
        self.nb.select(0)

    def download_weights(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Занято", "Сначала остановите процесс.")
            return
        cmd = [sys.executable, "download_weights.py"]
        self._start_process(cmd, "SETUP", "Загрузка весов FaceNet...")

    def test_telegram(self) -> None:
        from src.faceid.alerts import MultiTelegramAlerter, TelegramAlerter as _TA
        token    = self.telegram_bot_token_var.get().strip()
        chat_id  = self.telegram_chat_id_var.get().strip()
        chat_id2 = self.telegram_chat_id2_var.get().strip()
        if not token or not chat_id:
            messagebox.showerror("Ошибка", "Укажите Bot Token и Chat ID 1.")
            return
        self.set_status("TELEGRAM", "Отправка тестового сообщения...")
        self.append_log(">>> тест Telegram-уведомления")
        alerters = [_TA(bot_token=token, chat_id=chat_id)]
        if chat_id2:
            alerters.append(_TA(bot_token=token, chat_id=chat_id2))
        multi = MultiTelegramAlerter(alerters=alerters)
        threading.Thread(
            target=self._send_test_telegram, args=(multi, len(alerters)), daemon=True,
        ).start()

    def _send_test_telegram(self, alerter: object, count: int) -> None:
        try:
            now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            alerter.send_html(  # type: ignore[union-attr]
                f"&#128994; <b>FaceID — тест уведомлений</b>\n\n"
                f"Подключение к боту успешно.\n"
                f"&#128100; Получателей: <b>{count}</b>\n"
                f"&#128337; <code>{now}</code>"
            )
        except Exception as exc:
            self.after(0, self.append_log, f"telegram тест: ошибка — {exc}")
            self.after(0, self.set_status, "ERROR", "Telegram ошибка")
            return
        self.after(0, self.append_log, f"telegram тест: отправлено ({count} чел.)")
        self.after(0, self.set_status, "TELEGRAM", f"Telegram: тест отправлен ({count})")

    # ── process management ─────────────────────────────────────────────────────
    def _start_process(self, cmd: list[str], mode: str, status: str) -> None:
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(Path(__file__).parent),
            )
        except Exception as exc:
            self.proc = None
            messagebox.showerror("Ошибка запуска", str(exc))
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
        tl = text.lower()
        if "allow" in tl:
            self.set_status("VERIFY", "Доступ разрешён")
        elif "deny_" in tl:
            self.set_status("VERIFY", "Доступ отклонён")
        elif "error" in tl or "traceback" in tl:
            self.set_status("ERROR", "Проверьте журнал")

    def _on_process_exit(self, exit_code: int) -> None:
        self.append_log(f"[процесс завершился с кодом {exit_code}]")
        self.proc = None
        if exit_code == 0:
            self.set_status("IDLE", "Ожидание")
        else:
            self.set_status("ERROR", f"Ошибка (код {exit_code})")

    def stop_process(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        self.append_log("[запрошена остановка]")
        self.set_status("STOPPING", "Остановка...")

    # ── config persistence ─────────────────────────────────────────────────────
    _CONFIG_PATH = Path("artifacts/config.json")

    # Map of config key → (StringVar or BooleanVar attribute name)
    _CONFIG_VARS: list[str] = [
        "camera_index_var", "db_path_var", "device_var", "weights_path_var",
        "anti_spoof_model_path_var",
        "samples_var", "threshold_var",
        "liveness_mode_var", "liveness_enabled_var", "show_score_var",
        "event_ttl_days_var", "event_cooldown_var",
        "fast_window_var", "fast_face_jitter_var", "fast_ear_delta_var",
        "fast_require_blink_var",
        "passive_residual_var", "passive_texture_var", "passive_max_jitter_var",
        "anti_spoof_enabled_var", "anti_spoof_threshold_var",
        "relay_enabled_var", "relay_pin_var", "relay_duration_var", "relay_active_high_var",
        "telegram_enabled_var", "telegram_bot_token_var",
        "telegram_chat_id_var", "telegram_chat_id2_var",
        "telegram_fail_threshold_var", "telegram_window_var",
        "telegram_alert_cooldown_var", "telegram_unknown_cooldown_var",
        "telegram_send_photo_var", "telegram_admin_buttons_var",
        "telegram_admin_allow_sec_var",
        "key_path_var",
    ]

    def _load_config(self) -> None:
        import json as _json
        try:
            data = _json.loads(self._CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        for attr in self._CONFIG_VARS:
            var = getattr(self, attr, None)
            if var is not None and attr in data:
                try:
                    var.set(data[attr])
                except Exception:
                    pass

    def _save_config(self) -> None:
        import json as _json
        data: dict = {}
        for attr in self._CONFIG_VARS:
            var = getattr(self, attr, None)
            if var is not None:
                try:
                    data[attr] = var.get()
                except Exception:
                    pass
        try:
            self._CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._CONFIG_PATH.write_text(
                _json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_config()
        if self.proc is not None:
            self.proc.terminate()
        self.destroy()

    # ── data refresh ───────────────────────────────────────────────────────────
    def _refresh_events(self) -> None:
        db_path = self.db_path_var.get().strip()
        rows: list = []
        if db_path:
            try:
                conn = sqlite3.connect(db_path)
                try:
                    rows = conn.execute(
                        "SELECT ts_utc, decision, identity, score "
                        "FROM access_events ORDER BY id DESC LIMIT 40"
                    ).fetchall()
                finally:
                    conn.close()
            except Exception:
                rows = []

        self.events_tree.delete(*self.events_tree.get_children())
        for ts_utc, decision, identity, score in rows:
            time_short  = str(ts_utc)[11:19] if len(str(ts_utc)) >= 19 else str(ts_utc)
            score_text  = "-" if score is None or float(score) < 0 else f"{float(score):.3f}"
            if decision == "ALLOW":
                tag = "allow"
            elif decision == "DENY_LIVENESS":
                tag = "live"
            elif decision and decision.startswith("DENY"):
                tag = "deny"
            else:
                tag = "other"
            self.events_tree.insert("", "end",
                values=(time_short, decision, identity, score_text), tags=(tag,))

        self.after(self.events_refresh_ms, self._refresh_events)

    def _refresh_db_tab(self) -> None:
        from src.faceid.modeling import get_attendance_today_utc

        db_path = self.db_path_var.get().strip()
        if not db_path or not Path(db_path).exists():
            return
        try:
            conn = sqlite3.connect(db_path)
            try:
                users = conn.execute(
                    "SELECT name, updated_at FROM identities ORDER BY name"
                ).fetchall()
                today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
                row = conn.execute(
                    """SELECT
                        SUM(decision='ALLOW'),
                        SUM(decision LIKE 'DENY%'),
                        SUM(is_suspicious)
                       FROM access_events WHERE ts_utc >= ?""",
                    (today,),
                ).fetchone()
            finally:
                conn.close()
        except Exception:
            return

        # registered users
        self.users_tree.delete(*self.users_tree.get_children())
        for name, updated_at in users:
            date = str(updated_at)[:10] if updated_at else "—"
            self.users_tree.insert("", "end", values=(name, date))

        # today's attendance — one row per unique visitor, sorted by arrival
        attendance = get_attendance_today_utc(Path(db_path))
        self.attend_tree.delete(*self.attend_tree.get_children())
        for name, first_time in attendance:
            self.attend_tree.insert("", "end", values=(name, first_time),
                                    tags=("present",))
        self.attend_tree.tag_configure("present", foreground=C_SUCCESS)

        ok, denied, sus = (int(v or 0) for v in (row or (0, 0, 0)))
        self.stats_visitors_var.set(str(len(attendance)))
        self.stats_allow_var.set(str(ok))
        self.stats_deny_var.set(str(denied))
        self.stats_sus_var.set(str(sus))
        self.stats_total_var.set(str(ok + denied))

    def _delete_identity(self) -> None:
        sel = self.users_tree.selection()
        if not sel:
            messagebox.showinfo("Выбор", "Выберите пользователя для удаления.")
            return
        name = self.users_tree.item(sel[0])["values"][0]
        if not messagebox.askyesno("Удаление", f"Удалить пользователя «{name}»?"):
            return
        db_path = self.db_path_var.get().strip()
        try:
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("DELETE FROM identities WHERE name = ?", (name,))
                conn.commit()
            finally:
                conn.close()
            self.append_log(f"Пользователь «{name}» удалён из базы данных.")
            self._refresh_db_tab()
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _export_csv(self) -> None:
        """Export access events to a CSV file (ТЗ §4.1.6)."""
        from src.faceid.modeling import export_events_csv

        db_raw = self.db_path_var.get().strip()
        if not db_raw or not Path(db_raw).exists():
            messagebox.showerror("Ошибка", "База данных не найдена.")
            return
        out = filedialog.asksaveasfilename(
            title="Экспорт журнала событий",
            defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
            initialfile="access_events.csv",
        )
        if not out:
            return
        try:
            n = export_events_csv(Path(db_raw), Path(out))
            self.append_log(f"Экспорт завершён: {n} записей → {out}")
            messagebox.showinfo("Готово", f"Экспортировано {n} записей.")
        except Exception as exc:
            messagebox.showerror("Ошибка экспорта", str(exc))

    def _backup_db(self) -> None:
        """Plain SQLite backup (ТЗ §4.1.4 alt-flow)."""
        from src.faceid.modeling import backup_db

        db_raw = self.db_path_var.get().strip()
        if not db_raw or not Path(db_raw).exists():
            messagebox.showerror("Ошибка", "База данных не найдена.")
            return
        out = filedialog.asksaveasfilename(
            title="Резервная копия базы данных",
            defaultextension=".db",
            filetypes=[("SQLite базы данных", "*.db"), ("Все файлы", "*.*")],
            initialfile="faceid_backup.db",
        )
        if not out:
            return
        try:
            backup_db(Path(db_raw), Path(out))
            self.append_log(f"Резервная копия сохранена: {out}")
            messagebox.showinfo("Готово", "Резервная копия создана.")
        except Exception as exc:
            messagebox.showerror("Ошибка резервного копирования", str(exc))

    def _backup_db_encrypted(self) -> None:
        """Encrypted backup using Fernet (ТЗ §4.5 cryptography, §4.7 data protection)."""
        from src.faceid.crypto import backup_encrypted, ensure_key

        db_raw = self.db_path_var.get().strip()
        if not db_raw or not Path(db_raw).exists():
            messagebox.showerror("Ошибка", "База данных не найдена.")
            return
        out = filedialog.asksaveasfilename(
            title="Зашифрованная резервная копия",
            defaultextension=".enc",
            filetypes=[("Зашифрованные файлы", "*.enc"), ("Все файлы", "*.*")],
            initialfile="faceid_backup.db.enc",
        )
        if not out:
            return
        key_raw = self.key_path_var.get().strip()
        key_path = Path(key_raw) if key_raw else Path("artifacts/faceid.key")
        try:
            key = ensure_key(key_path)
            backup_encrypted(Path(db_raw), Path(out), key)
            self.append_log(
                f"Зашифрованная копия сохранена: {out}  (ключ: {key_path})"
            )
            messagebox.showinfo(
                "Готово",
                f"Зашифрованная копия создана.\n\nКлюч шифрования:\n{key_path}\n\n"
                "Сохраните ключ — без него расшифровка невозможна.",
            )
        except Exception as exc:
            messagebox.showerror("Ошибка шифрования", str(exc))


def main() -> None:
    app = FaceIdGui()
    app.mainloop()


if __name__ == "__main__":
    main()
