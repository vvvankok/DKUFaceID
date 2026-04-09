from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

from src.faceid.alerts import TelegramAlerter

# ─────────────────── Цветовая палитра ────────────────────────────────────────
C_BG        = "#f1f5f9"
C_HEADER    = "#0f172a"
C_ACCENT    = "#2563eb"
C_ACCENT2   = "#1d4ed8"
C_SUCCESS   = "#16a34a"
C_ERROR     = "#dc2626"
C_WARN      = "#d97706"
C_WHITE     = "#ffffff"
C_BORDER    = "#e2e8f0"
C_TEXT      = "#0f172a"
C_MUTED     = "#64748b"
C_LOG_BG    = "#0f172a"
C_LOG_FG    = "#cbd5e1"

ROW_ALLOW = {"bg": "#dcfce7", "fg": "#14532d"}
ROW_DENY  = {"bg": "#fee2e2", "fg": "#7f1d1d"}
ROW_LIVE  = {"bg": "#ffedd5", "fg": "#7c2d12"}
ROW_OTHER = {"bg": "#f1f5f9", "fg": "#334155"}


class FaceIdGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("FaceID — Система контроля доступа")
        self.geometry("1140x760")
        self.minsize(980, 660)
        self.configure(bg=C_BG)

        self.proc: subprocess.Popen[str] | None = None
        self.output_thread: threading.Thread | None = None

        self._init_vars()
        self._load_config()
        self._setup_style()
        self._build_ui()
        self._refresh_events()
        self._refresh_db_tab()
        self._refresh_model_tab()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ═══════════════════════════════ ПЕРЕМЕННЫЕ ═══════════════════════════════
    def _init_vars(self) -> None:
        # окружение
        self.camera_index_var          = tk.StringVar(value="0")
        self.db_path_var               = tk.StringVar(value="artifacts/faceid_identities.db")
        self.weights_path_var          = tk.StringVar(value="")
        self.device_var                = tk.StringVar(value="")

        # регистрация
        self.name_var                  = tk.StringVar(value="")
        self.samples_var               = tk.StringVar(value="20")

        # верификация
        self.threshold_var             = tk.StringVar(value="0.72")
        self.show_score_var            = tk.BooleanVar(value=True)
        self.liveness_enabled_var      = tk.BooleanVar(value=True)
        self.anti_spoof_enabled_var    = tk.BooleanVar(value=True)
        self.anti_spoof_model_path_var = tk.StringVar(value="artifacts/anti_spoof_model.pt")
        self.anti_spoof_threshold_var  = tk.StringVar(value="0.80")
        self.event_ttl_days_var        = tk.StringVar(value="7")
        self.event_cooldown_var        = tk.StringVar(value="1.2")
        self.passive_residual_var      = tk.StringVar(value="0.24")
        self.passive_texture_var       = tk.StringVar(value="45")
        self.passive_max_jitter_var    = tk.StringVar(value="8")

        # реле
        self.relay_enabled_var         = tk.BooleanVar(value=False)
        self.relay_pin_var             = tk.StringVar(value="18")
        self.relay_duration_var        = tk.StringVar(value="2.0")
        self.relay_active_high_var     = tk.BooleanVar(value=True)

        # telegram
        self.telegram_enabled_var           = tk.BooleanVar(value=False)
        self.telegram_bot_token_var         = tk.StringVar(value="")
        self.telegram_chat_id_var           = tk.StringVar(value="")
        self.telegram_chat_id2_var          = tk.StringVar(value="")
        self.telegram_fail_threshold_var    = tk.StringVar(value="3")
        self.telegram_window_var            = tk.StringVar(value="20")
        self.telegram_alert_cooldown_var    = tk.StringVar(value="40")
        self.telegram_unknown_cooldown_var  = tk.StringVar(value="20")
        self.telegram_send_photo_var        = tk.BooleanVar(value=True)
        self.telegram_admin_buttons_var     = tk.BooleanVar(value=True)
        self.telegram_admin_allow_sec_var   = tk.StringVar(value="15")

        # шифрование
        self.key_path_var              = tk.StringVar(value="artifacts/faceid.key")

        # модель антиспуфинга
        self.capture_label_var         = tk.StringVar(value="live")
        self.capture_person_var        = tk.StringVar(value="session1")
        self.capture_count_var         = tk.StringVar(value="150")
        self.model_dataset_dir_var     = tk.StringVar(value="dataset/anti_spoof")
        self.train_epochs_var          = tk.StringVar(value="8")

        # статус
        self.status_var                = tk.StringVar(value="Ожидание")
        self.mode_var                  = tk.StringVar(value="  IDLE  ")
        self.events_refresh_ms         = 1500

        # метрики модели (отображение)
        self.model_acc_var             = tk.StringVar(value="—")
        self.model_live_recall_var     = tk.StringVar(value="—")
        self.model_spoof_recall_var    = tk.StringVar(value="—")
        self.model_dataset_info_var    = tk.StringVar(value="—")
        self.dataset_live_var          = tk.StringVar(value="—")
        self.dataset_spoof_var         = tk.StringVar(value="—")

        # дашборд
        self.stats_visitors_var        = tk.StringVar(value="—")
        self.stats_allow_var           = tk.StringVar(value="—")
        self.stats_deny_var            = tk.StringVar(value="—")
        self.stats_sus_var             = tk.StringVar(value="—")
        self.stats_registered_var      = tk.StringVar(value="—")
        self.last_refresh_var          = tk.StringVar(value="")

        # PIN-защита настроек
        self.pin_enabled_var           = tk.BooleanVar(value=False)
        self.pin_hash_var              = tk.StringVar(value="")

        # ночной порог
        self.threshold_after_hours_var  = tk.BooleanVar(value=False)
        self.threshold_night_var        = tk.StringVar(value="0.82")
        self.threshold_night_start_var  = tk.StringVar(value="20")
        self.threshold_night_end_var    = tk.StringVar(value="8")

        # отчёт за период
        today = datetime.date.today().isoformat()
        self.report_start_var          = tk.StringVar(value=today)
        self.report_end_var            = tk.StringVar(value=today)

        # расписание / clustering
        self.schedule_enabled_var      = tk.BooleanVar(value=False)
        self.cluster_unknowns_var      = tk.BooleanVar(value=True)

    # ═══════════════════════════════ СТИЛИ ════════════════════════════════════
    def _setup_style(self) -> None:
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=C_BG, foreground=C_TEXT, font=("Segoe UI", 10))
        s.configure("TFrame", background=C_BG)
        s.configure("TLabel", background=C_BG, foreground=C_TEXT)
        s.configure("TCheckbutton", background=C_BG, foreground=C_TEXT)
        s.configure("TLabelframe", background=C_BG, bordercolor=C_BORDER, relief="flat")
        s.configure("TLabelframe.Label", background=C_BG, foreground=C_ACCENT,
                    font=("Segoe UI", 10, "bold"))
        s.configure("TEntry", fieldbackground=C_WHITE)
        s.configure("TCombobox", fieldbackground=C_WHITE)
        s.configure("TScrollbar", background=C_BG, troughcolor=C_BORDER)
        # notebook
        s.configure("TNotebook", background=C_BG, tabmargins=[2, 5, 2, 0])
        s.configure("TNotebook.Tab", background=C_BORDER, foreground=C_MUTED,
                    padding=[18, 9], font=("Segoe UI", 10))
        s.map("TNotebook.Tab",
              background=[("selected", C_WHITE)],
              foreground=[("selected", C_ACCENT)],
              font=[("selected", ("Segoe UI", 10, "bold"))])
        # кнопки
        for name, bg, hover in [
            ("TButton",          C_ACCENT,   C_ACCENT2),
            ("Success.TButton",  C_SUCCESS,  "#15803d"),
            ("Danger.TButton",   C_ERROR,    "#b91c1c"),
            ("Neutral.TButton",  "#64748b",  "#475569"),
            ("Warn.TButton",     C_WARN,     "#b45309"),
        ]:
            s.configure(name, background=bg, foreground=C_WHITE, relief="flat",
                        padding=[14, 7], font=("Segoe UI", 10, "bold"))
            s.map(name, background=[("active", hover), ("disabled", C_BORDER)],
                  foreground=[("disabled", C_MUTED)])
        # treeview
        s.configure("Treeview", background=C_WHITE, fieldbackground=C_WHITE,
                    foreground=C_TEXT, font=("Segoe UI", 10), rowheight=28)
        s.configure("Treeview.Heading", background="#1e293b", foreground=C_WHITE,
                    font=("Segoe UI", 10, "bold"), relief="flat")
        s.map("Treeview",
              background=[("selected", C_ACCENT)],
              foreground=[("selected", C_WHITE)])
        s.configure("Muted.TLabel", foreground=C_MUTED, font=("Segoe UI", 9))
        s.configure("Card.TFrame", background=C_WHITE, relief="flat")
        s.configure("Big.TLabel", font=("Segoe UI", 24, "bold"), background=C_WHITE)
        s.configure("BigMuted.TLabel", font=("Segoe UI", 11), background=C_WHITE,
                    foreground=C_MUTED)
        s.configure("Section.TLabel", font=("Segoe UI", 11, "bold"), foreground=C_TEXT)

    # ═══════════════════════════════ ВЕРХНИЙ УРОВЕНЬ ══════════════════════════
    def _build_ui(self) -> None:
        # ── шапка ─────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C_HEADER, height=56)
        hdr.pack(fill=tk.X, side=tk.TOP)
        hdr.pack_propagate(False)

        tk.Label(hdr, text="  FaceID", bg=C_HEADER, fg=C_WHITE,
                 font=("Segoe UI", 17, "bold")).pack(side=tk.LEFT, padx=(14, 6), pady=10)
        tk.Label(hdr, text="Казахстанско-Немецкий Университет",
                 bg=C_HEADER, fg="#94a3b8", font=("Segoe UI", 10)).pack(side=tk.LEFT)

        # статус-пилюля справа
        self._status_pill = tk.Label(
            hdr, textvariable=self.mode_var,
            bg="#374151", fg="#f9fafb", font=("Segoe UI", 10, "bold"), padx=12, pady=3)
        self._status_pill.pack(side=tk.RIGHT, padx=14, pady=14)
        tk.Label(hdr, textvariable=self.status_var, bg=C_HEADER, fg="#94a3b8",
                 font=("Segoe UI", 10)).pack(side=tk.RIGHT, padx=(0, 4))

        # кнопка "Стоп"
        ttk.Button(hdr, text="⏹  Стоп", style="Danger.TButton",
                   command=self.stop_process).pack(side=tk.RIGHT, padx=(0, 8), pady=10)

        # ── вкладки ───────────────────────────────────────────────────────────
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 6))

        self._tab_main      = ttk.Frame(self.nb, padding=12)
        self._tab_employees = ttk.Frame(self.nb, padding=12)
        self._tab_journal   = ttk.Frame(self.nb, padding=12)
        self._tab_model     = ttk.Frame(self.nb, padding=12)
        self._tab_settings  = ttk.Frame(self.nb, padding=12)
        self._tab_analytics = ttk.Frame(self.nb, padding=12)

        self.nb.add(self._tab_main,      text="  Главная  ")
        self.nb.add(self._tab_employees, text="  Сотрудники  ")
        self.nb.add(self._tab_journal,   text="  Журнал доступа  ")
        self.nb.add(self._tab_model,     text="  Модель  ")
        self.nb.add(self._tab_settings,  text="  Настройки  ")
        self.nb.add(self._tab_analytics, text="  Аналитика  ")

        self._build_tab_main()
        self._build_tab_employees()
        self._build_tab_journal()
        self._build_tab_model()
        self._build_tab_settings()
        self._build_tab_analytics()

        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self._auto_refresh_stats()

    # ═══════════════════════════ TAB: ГЛАВНАЯ ═════════════════════════════════
    def _build_tab_main(self) -> None:
        t = self._tab_main
        t.columnconfigure(0, weight=3)
        t.columnconfigure(1, weight=2)
        t.rowconfigure(2, weight=1)

        # ── панель управления (кнопки + статус) ───────────────────────────────
        ctrl = tk.Frame(t, bg=C_WHITE, relief="flat",
                        highlightbackground=C_BORDER, highlightthickness=1)
        ctrl.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ctrl.columnconfigure(2, weight=1)

        ttk.Button(ctrl, text="▶  Запустить систему", style="Success.TButton",
                   command=self.start_verify).pack(side=tk.LEFT, padx=14, pady=12)
        ttk.Button(ctrl, text="⏹  Остановить", style="Danger.TButton",
                   command=self.stop_process).pack(side=tk.LEFT, padx=(0, 14), pady=12)

        tk.Frame(ctrl, bg=C_BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y, pady=8)

        ttk.Button(ctrl, text="➕  Зарегистрировать", style="TButton",
                   command=lambda: self.nb.select(1)).pack(side=tk.LEFT, padx=14, pady=12)
        ttk.Button(ctrl, text="Скачать веса", style="Neutral.TButton",
                   command=self.download_weights).pack(side=tk.LEFT, pady=12)

        tk.Label(ctrl, textvariable=self.last_refresh_var, bg=C_WHITE,
                 fg=C_MUTED, font=("Segoe UI", 9)).pack(side=tk.RIGHT, padx=14)

        # ── карточки статистики ───────────────────────────────────────────────
        cards = tk.Frame(t, bg=C_BG)
        cards.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        for i in range(5):
            cards.columnconfigure(i, weight=1)

        self._stat_card(cards, 0, "Зарегистрировано",  self.stats_registered_var, C_ACCENT)
        self._stat_card(cards, 1, "Пришли сегодня",    self.stats_visitors_var,   "#7c3aed")
        self._stat_card(cards, 2, "Разрешений",        self.stats_allow_var,      C_SUCCESS)
        self._stat_card(cards, 3, "Отказов",           self.stats_deny_var,       C_ERROR)
        self._stat_card(cards, 4, "Подозрительных",    self.stats_sus_var,        C_WARN)

        # ── журнал вывода ─────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(t, text="Системный журнал", padding=6)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log = tk.Text(log_frame, wrap=tk.WORD, bg=C_LOG_BG, fg=C_LOG_FG,
                           insertbackground=C_LOG_FG, relief=tk.FLAT,
                           font=("Consolas", 10), state=tk.DISABLED)
        log_sb = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=log_sb.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        log_sb.grid(row=0, column=1, sticky="ns")

        # ── последние события ─────────────────────────────────────────────────
        ev_frame = ttk.LabelFrame(t, text="Последние события", padding=6)
        ev_frame.grid(row=2, column=1, sticky="nsew")
        ev_frame.rowconfigure(0, weight=1)
        ev_frame.columnconfigure(0, weight=1)

        cols = ("time", "decision", "identity", "score")
        self.events_tree = ttk.Treeview(ev_frame, columns=cols, show="headings", height=18)
        self.events_tree.heading("time",     text="Время")
        self.events_tree.heading("decision", text="Решение")
        self.events_tree.heading("identity", text="Сотрудник")
        self.events_tree.heading("score",    text="Балл")
        self.events_tree.column("time",     width=72,  anchor="center")
        self.events_tree.column("decision", width=110, anchor="w")
        self.events_tree.column("identity", width=100, anchor="w")
        self.events_tree.column("score",    width=60,  anchor="center")
        for tag, cfg in [("allow", ROW_ALLOW), ("deny", ROW_DENY),
                          ("live", ROW_LIVE),  ("other", ROW_OTHER)]:
            self.events_tree.tag_configure(tag, background=cfg["bg"], foreground=cfg["fg"])
        ev_sb = ttk.Scrollbar(ev_frame, orient=tk.VERTICAL, command=self.events_tree.yview)
        self.events_tree.configure(yscrollcommand=ev_sb.set)
        self.events_tree.grid(row=0, column=0, sticky="nsew")
        ev_sb.grid(row=0, column=1, sticky="ns")

    def _stat_card(self, parent: tk.Frame, col: int,
                   title: str, var: tk.StringVar, color: str) -> None:
        card = tk.Frame(parent, bg=C_WHITE,
                        highlightbackground=C_BORDER, highlightthickness=1)
        card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 5, 0), pady=2)
        tk.Label(card, text=title, bg=C_WHITE, fg=C_MUTED,
                 font=("Segoe UI", 9)).pack(pady=(10, 2), padx=14)
        tk.Label(card, textvariable=var, bg=C_WHITE, fg=color,
                 font=("Segoe UI", 26, "bold")).pack(pady=(0, 10), padx=14)

    # ═══════════════════════════ TAB: СОТРУДНИКИ ══════════════════════════════
    def _build_tab_employees(self) -> None:
        t = self._tab_employees
        t.columnconfigure(0, weight=1)
        t.columnconfigure(1, weight=1)
        t.rowconfigure(1, weight=1)

        # ── заголовок ─────────────────────────────────────────────────────────
        ttk.Label(t, text="Управление сотрудниками", style="Section.TLabel"
                  ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        # ── список пользователей ──────────────────────────────────────────────
        left = ttk.LabelFrame(t, text="Зарегистрированные сотрудники", padding=10)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        btn_row = ttk.Frame(left)
        btn_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Button(btn_row, text="Обновить", style="Neutral.TButton",
                   command=self._refresh_db_tab).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Удалить выбранного", style="Danger.TButton",
                   command=self._delete_identity).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Расписание", style="Neutral.TButton",
                   command=self._show_schedule_dialog).pack(side=tk.LEFT)

        ucols = ("name", "updated")
        self.users_tree = ttk.Treeview(left, columns=ucols, show="headings")
        self.users_tree.heading("name",    text="Имя / ФИО")
        self.users_tree.heading("updated", text="Дата добавления")
        self.users_tree.column("name",    width=160, anchor="w")
        self.users_tree.column("updated", width=120, anchor="center")
        u_sb = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.users_tree.yview)
        self.users_tree.configure(yscrollcommand=u_sb.set)
        self.users_tree.grid(row=1, column=0, sticky="nsew")
        u_sb.grid(row=1, column=1, sticky="ns")
        self.users_tree.bind("<<TreeviewSelect>>", self._on_user_select)

        # фото сотрудника
        self._photo_label = tk.Label(left, bg=C_WHITE, relief="flat",
                                     highlightbackground=C_BORDER, highlightthickness=1)
        self._photo_label.grid(row=2, column=0, columnspan=2, pady=(8, 0), sticky="w")
        self._photo_ref = None  # keep reference to avoid GC

        # ── форма регистрации ─────────────────────────────────────────────────
        right = ttk.LabelFrame(t, text="Добавить нового сотрудника", padding=14)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(1, weight=1)

        self._field(right, 0, "ФИО / имя",       self.name_var)
        self._field(right, 1, "Число снимков",   self.samples_var)

        ttk.Label(right, text=(
            "Рекомендации для точного распознавания:\n\n"
            "  • Равномерное освещение без теней на лице\n"
            "  • Расстояние 40–70 см до камеры\n"
            "  • Слегка поворачивайте голову при съёмке\n"
            "  • Уберите аксессуары, закрывающие лицо\n"
            "  • Рекомендуется 20–30 снимков"
        ), style="Muted.TLabel", justify=tk.LEFT).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(14, 18))

        ttk.Button(right, text="▶  Начать регистрацию",
                   command=self.start_register).grid(
            row=3, column=0, columnspan=2, sticky="ew")

        ttk.Label(right, text=(
            "Камера откроется автоматически.\n"
            "После съёмки сотрудник сразу добавляется в базу."
        ), style="Muted.TLabel", justify=tk.LEFT).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

    # ═══════════════════════════ TAB: ЖУРНАЛ ══════════════════════════════════
    def _build_tab_journal(self) -> None:
        t = self._tab_journal
        t.columnconfigure(0, weight=3)
        t.columnconfigure(1, weight=2)
        t.rowconfigure(2, weight=1)

        # ── кнопки ────────────────────────────────────────────────────────────
        btn_row = ttk.Frame(t)
        btn_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(btn_row, text="Обновить", style="Neutral.TButton",
                   command=self._refresh_db_tab).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Экспорт в CSV", style="Neutral.TButton",
                   command=self._export_csv).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Экспорт PDF", style="TButton",
                   command=self._export_pdf).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Резервная копия", style="Neutral.TButton",
                   command=self._backup_db).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Зашифрованная копия", style="Neutral.TButton",
                   command=self._backup_db_encrypted).pack(side=tk.LEFT)

        # ── отчёт за период ───────────────────────────────────────────────────
        rep_row = tk.Frame(t, bg=C_BG)
        rep_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        tk.Label(rep_row, text="Отчёт за период:", bg=C_BG, fg=C_TEXT,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(rep_row, text="с", bg=C_BG, fg=C_MUTED).pack(side=tk.LEFT)
        ttk.Entry(rep_row, textvariable=self.report_start_var, width=11).pack(
            side=tk.LEFT, padx=(4, 8))
        tk.Label(rep_row, text="по", bg=C_BG, fg=C_MUTED).pack(side=tk.LEFT)
        ttk.Entry(rep_row, textvariable=self.report_end_var, width=11).pack(
            side=tk.LEFT, padx=(4, 8))
        ttk.Button(rep_row, text="Скачать CSV", style="TButton",
                   command=self._export_attendance_report).pack(side=tk.LEFT)
        ttk.Button(rep_row, text="Отправить PDF в Telegram", style="Neutral.TButton",
                   command=self._send_pdf_to_telegram).pack(side=tk.LEFT, padx=(8, 0))
        tk.Label(rep_row, text="(формат ГГГГ-ММ-ДД)", bg=C_BG,
                 fg=C_MUTED, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(8, 0))

        # ── журнал событий ────────────────────────────────────────────────────
        ev_card = ttk.LabelFrame(t, text="Журнал событий доступа", padding=8)
        ev_card.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        ev_card.rowconfigure(0, weight=1)
        ev_card.columnconfigure(0, weight=1)

        jcols = ("time", "decision", "identity", "score", "reason")
        self.journal_tree = ttk.Treeview(ev_card, columns=jcols, show="headings")
        self.journal_tree.heading("time",     text="Время (UTC)")
        self.journal_tree.heading("decision", text="Решение")
        self.journal_tree.heading("identity", text="Сотрудник")
        self.journal_tree.heading("score",    text="Балл")
        self.journal_tree.heading("reason",   text="Причина")
        self.journal_tree.column("time",     width=140, anchor="center")
        self.journal_tree.column("decision", width=120, anchor="w")
        self.journal_tree.column("identity", width=130, anchor="w")
        self.journal_tree.column("score",    width=70,  anchor="center")
        self.journal_tree.column("reason",   width=160, anchor="w")
        for tag, cfg in [("allow", ROW_ALLOW), ("deny", ROW_DENY),
                          ("live", ROW_LIVE),  ("other", ROW_OTHER)]:
            self.journal_tree.tag_configure(tag, background=cfg["bg"], foreground=cfg["fg"])
        j_sb = ttk.Scrollbar(ev_card, orient=tk.VERTICAL, command=self.journal_tree.yview)
        self.journal_tree.configure(yscrollcommand=j_sb.set)
        self.journal_tree.grid(row=0, column=0, sticky="nsew")
        j_sb.grid(row=0, column=1, sticky="ns")

        # ── посещаемость ──────────────────────────────────────────────────────
        att_card = ttk.LabelFrame(t, text="Посещаемость сегодня", padding=8)
        att_card.grid(row=2, column=1, sticky="nsew")
        att_card.rowconfigure(1, weight=1)
        att_card.columnconfigure(0, weight=1)

        # мини-статистика сверху
        stats_row = tk.Frame(att_card, bg=C_BG)
        stats_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        def _mini(parent, label, var, color):
            f = tk.Frame(parent, bg=C_WHITE,
                         highlightbackground=C_BORDER, highlightthickness=1)
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
            tk.Label(f, text=label, bg=C_WHITE, fg=C_MUTED,
                     font=("Segoe UI", 8)).pack(pady=(6, 0))
            tk.Label(f, textvariable=var, bg=C_WHITE, fg=color,
                     font=("Segoe UI", 16, "bold")).pack(pady=(0, 6))

        self.stats_allow_var  = tk.StringVar(value="—")
        self.stats_deny_var   = tk.StringVar(value="—")
        self.stats_sus_var    = tk.StringVar(value="—")
        self.stats_total_var  = tk.StringVar(value="—")
        _mini(stats_row, "Разрешено",      self.stats_allow_var,  C_SUCCESS)
        _mini(stats_row, "Отказано",       self.stats_deny_var,   C_ERROR)
        _mini(stats_row, "Подозрительных", self.stats_sus_var,    C_WARN)
        _mini(stats_row, "Всего",          self.stats_total_var,  C_MUTED)

        acols = ("name", "arrived")
        self.attend_tree = ttk.Treeview(att_card, columns=acols, show="headings")
        self.attend_tree.heading("name",    text="Сотрудник")
        self.attend_tree.heading("arrived", text="Время входа")
        self.attend_tree.column("name",    width=130, anchor="w")
        self.attend_tree.column("arrived", width=90,  anchor="center")
        a_sb = ttk.Scrollbar(att_card, orient=tk.VERTICAL, command=self.attend_tree.yview)
        self.attend_tree.configure(yscrollcommand=a_sb.set)
        self.attend_tree.grid(row=1, column=0, sticky="nsew")
        a_sb.grid(row=1, column=1, sticky="ns")
        self.attend_tree.tag_configure("present", foreground=C_SUCCESS,
                                       font=("Segoe UI", 10, "bold"))

    # ═══════════════════════════ TAB: МОДЕЛЬ ══════════════════════════════════
    def _build_tab_model(self) -> None:
        t = self._tab_model
        t.columnconfigure(0, weight=1)
        t.columnconfigure(1, weight=1)
        t.rowconfigure(2, weight=1)

        ttk.Label(t, text="Управление моделью защиты от подделок",
                  style="Section.TLabel").grid(row=0, column=0, columnspan=2,
                                               sticky="w", pady=(0, 10))

        # ── карточка текущей модели ───────────────────────────────────────────
        info = ttk.LabelFrame(t, text="Текущая модель антиспуфинга", padding=14)
        info.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        info.columnconfigure(0, weight=1)
        info.columnconfigure(1, weight=1)
        info.columnconfigure(2, weight=1)
        info.columnconfigure(3, weight=1)

        def _metric(parent, col, title, var, color):
            f = tk.Frame(parent, bg=C_WHITE,
                         highlightbackground=C_BORDER, highlightthickness=1)
            f.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0), pady=4)
            tk.Label(f, text=title, bg=C_WHITE, fg=C_MUTED,
                     font=("Segoe UI", 9)).pack(pady=(10, 2), padx=16)
            tk.Label(f, textvariable=var, bg=C_WHITE, fg=color,
                     font=("Segoe UI", 20, "bold")).pack(pady=(0, 10), padx=16)

        _metric(info, 0, "Точность модели",      self.model_acc_var,         C_SUCCESS)
        _metric(info, 1, "Живых не пропустила",  self.model_live_recall_var,  C_ACCENT)
        _metric(info, 2, "Подделок поймала",      self.model_spoof_recall_var, "#7c3aed")
        _metric(info, 3, "Датасет обучения",      self.model_dataset_info_var, C_MUTED)

        ttk.Button(info, text="Обновить", style="Neutral.TButton",
                   command=self._refresh_model_tab).grid(
            row=1, column=0, columnspan=4, sticky="e", pady=(4, 0))

        # ── захват данных ─────────────────────────────────────────────────────
        cap_frame = ttk.LabelFrame(t, text="Захват обучающих данных", padding=14)
        cap_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 8))
        cap_frame.columnconfigure(1, weight=1)

        ttk.Label(cap_frame, text=(
            "Для дообучения модели нужно снять примеры\n"
            "живых лиц и фото-подделок (фото/телефон)."
        ), style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        ttk.Label(cap_frame, text="Тип снимков").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=6)
        ttk.Combobox(cap_frame, textvariable=self.capture_label_var,
                     values=["live", "spoof"], state="readonly").grid(
            row=1, column=1, sticky="ew", pady=6)

        ttk.Label(cap_frame, text=(
            "   live  — живое лицо перед камерой\n"
            "   spoof — фото/видео на экране"
        ), style="Muted.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self._field(cap_frame, 3, "Имя сессии",       self.capture_person_var)
        self._field(cap_frame, 4, "Кол-во снимков",   self.capture_count_var)
        self._field(cap_frame, 5, "Папка датасета",   self.model_dataset_dir_var, browse=False)

        ttk.Button(cap_frame, text="▶  Начать захват данных",
                   command=self.start_capture).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(16, 0))

        ttk.Label(cap_frame, text=(
            "Камера откроется автоматически.\n"
            "Нажмите Q для завершения съёмки."
        ), style="Muted.TLabel").grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # ── обучение модели ───────────────────────────────────────────────────
        train_frame = ttk.LabelFrame(t, text="Переобучение модели", padding=14)
        train_frame.grid(row=2, column=1, sticky="nsew")
        train_frame.columnconfigure(1, weight=1)

        ttk.Label(train_frame, text=(
            "После сбора данных запустите обучение.\n"
            "Новая модель заменит текущую."
        ), style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))

        # статистика датасета
        ds_card = tk.Frame(train_frame, bg=C_WHITE,
                           highlightbackground=C_BORDER, highlightthickness=1)
        ds_card.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))

        for col, (label, var) in enumerate([
            ("Живых снимков",  self.dataset_live_var),
            ("Снимков подделок", self.dataset_spoof_var),
        ]):
            tk.Label(ds_card, text=label, bg=C_WHITE, fg=C_MUTED,
                     font=("Segoe UI", 9)).grid(row=0, column=col, padx=20, pady=(8, 2))
            tk.Label(ds_card, textvariable=var, bg=C_WHITE, fg=C_TEXT,
                     font=("Segoe UI", 16, "bold")).grid(row=1, column=col, padx=20, pady=(0, 8))

        self._field(train_frame, 2, "Эпох обучения", self.train_epochs_var)
        self._field(train_frame, 3, "Устройство",    self.device_var)

        ttk.Button(train_frame, text="▶  Запустить обучение", style="Warn.TButton",
                   command=self.start_train_model).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(16, 0))

        ttk.Label(train_frame, text=(
            "Обучение занимает 3–10 минут на CPU.\n"
            "Во время обучения верификация недоступна.\n"
            "Новая модель сохраняется автоматически."
        ), style="Muted.TLabel").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

    # ═══════════════════════════ TAB: НАСТРОЙКИ ═══════════════════════════════
    def _build_tab_settings(self) -> None:
        t = self._tab_settings

        canvas = tk.Canvas(t, bg=C_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(t, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win_id, width=e.width))
        self.bind_all("<MouseWheel>",
                      lambda e: self._scroll_canvas_event(canvas, e))

        inner.columnconfigure(0, weight=1)
        inner.columnconfigure(1, weight=1)

        # ── основное окружение ────────────────────────────────────────────────
        env = ttk.LabelFrame(inner, text="Окружение системы", padding=12)
        env.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        env.columnconfigure(1, weight=1)
        self._field(env, 0, "Индекс камеры",           self.camera_index_var)
        self._field(env, 1, "База данных (.db)",        self.db_path_var, browse=True)
        self._field(env, 2, "Веса FaceNet (.pt)",       self.weights_path_var, browse=True)
        self._field(env, 3, "Устройство (cpu / cuda)",  self.device_var)
        ttk.Button(env, text="Скачать веса FaceNet", style="Neutral.TButton",
                   command=self.download_weights).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # ── параметры верификации ─────────────────────────────────────────────
        vp = ttk.LabelFrame(inner, text="Верификация", padding=12)
        vp.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 10))
        vp.columnconfigure(1, weight=1)
        self._field(vp, 0, "Порог схожести (0–1)",     self.threshold_var)
        self._field(vp, 1, "Антиспуф-модель (.pt)",    self.anti_spoof_model_path_var, browse=True)
        self._field(vp, 2, "Порог антиспуфа (0–1)",    self.anti_spoof_threshold_var)
        self._field(vp, 3, "Хранить события (дней)",   self.event_ttl_days_var)

        chk = ttk.Frame(vp)
        chk.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(chk, text="Живость (optical flow)",
                        variable=self.liveness_enabled_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(chk, text="Антиспуф-модель",
                        variable=self.anti_spoof_enabled_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(chk, text="Показывать балл",
                        variable=self.show_score_var).pack(side=tk.LEFT)

        chk2b = ttk.Frame(vp)
        chk2b.grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Checkbutton(chk2b, text="Расписание доступа",
                        variable=self.schedule_enabled_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(chk2b, text="Логировать неизвестных (кластеризация)",
                        variable=self.cluster_unknowns_var).pack(side=tk.LEFT)

        # ── реле / замок ──────────────────────────────────────────────────────
        rl = ttk.LabelFrame(inner, text="Управление реле (замок)", padding=12)
        rl.grid(row=1, column=1, sticky="nsew", pady=(0, 10))
        rl.columnconfigure(1, weight=1)
        ttk.Checkbutton(rl, text="Включить управление реле",
                        variable=self.relay_enabled_var).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self._field(rl, 1, "GPIO пин (BCM)",         self.relay_pin_var)
        self._field(rl, 2, "Время открытия (сек)",   self.relay_duration_var)
        ttk.Checkbutton(rl, text="Активный HIGH",
                        variable=self.relay_active_high_var).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(rl, text="На Raspberry Pi управляет реле-замком.\nНа ПК — симуляция в журнале.",
                  style="Muted.TLabel").grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # ── telegram ──────────────────────────────────────────────────────────
        tg = ttk.LabelFrame(inner, text="Telegram-уведомления", padding=12)
        tg.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        tg.columnconfigure(1, weight=1)

        ttk.Checkbutton(tg, text="Включить уведомления в Telegram",
                        variable=self.telegram_enabled_var).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(tg, text="Bot Token").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        token_entry = ttk.Entry(tg, textvariable=self.telegram_bot_token_var, show="●")
        token_entry.grid(row=1, column=1, sticky="ew", pady=4)
        self._token_visible = False
        def _toggle():
            self._token_visible = not self._token_visible
            token_entry.config(show="" if self._token_visible else "●")
            btn_eye.config(text="Скрыть" if self._token_visible else "Показать")
        btn_eye = ttk.Button(tg, text="Показать", style="Neutral.TButton", command=_toggle)
        btn_eye.grid(row=1, column=2, padx=(8, 0), pady=4)

        self._field(tg, 2, "Chat ID 1 (основной)",       self.telegram_chat_id_var)
        self._field(tg, 3, "Chat ID 2 (дополнительный)", self.telegram_chat_id2_var)
        self._field(tg, 4, "Порог тревоги (отказов)",    self.telegram_fail_threshold_var)
        self._field(tg, 5, "Кулдаун алертов (сек)",      self.telegram_alert_cooldown_var)
        self._field(tg, 6, "Разрешение через бота (сек)",self.telegram_admin_allow_sec_var)

        chk2 = ttk.Frame(tg)
        chk2.grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(chk2, text="Отправлять фото",
                        variable=self.telegram_send_photo_var).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Checkbutton(chk2, text="Кнопки Разрешить/Отклонить",
                        variable=self.telegram_admin_buttons_var).pack(side=tk.LEFT)

        ttk.Label(tg, text=(
            "Команды бота: /help  /list  /stats  /attendance  /status"
        ), style="Muted.TLabel").grid(row=8, column=0, columnspan=3, sticky="w", pady=(8, 0))

        ttk.Button(tg, text="Отправить тестовое сообщение", style="Neutral.TButton",
                   command=self.test_telegram).grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(10, 0))

        # ── расширенные ───────────────────────────────────────────────────────
        adv = ttk.LabelFrame(inner, text="Расширенные параметры", padding=12)
        adv.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        adv.columnconfigure(1, weight=1)
        adv.columnconfigure(3, weight=1)

        ttk.Label(adv, text="Пассивная живость", style="Muted.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        ttk.Label(adv, text="Остаточный поток").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(adv, textvariable=self.passive_residual_var).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(adv, text="Мин. текстура").grid(row=1, column=2, sticky="w", padx=(16, 8), pady=3)
        ttk.Entry(adv, textvariable=self.passive_texture_var).grid(row=1, column=3, sticky="ew", pady=3)
        ttk.Label(adv, text="Макс. джиттер (px)").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(adv, textvariable=self.passive_max_jitter_var).grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Label(adv, text="Кулдаун событий (сек)").grid(row=2, column=2, sticky="w", padx=(16, 8), pady=3)
        ttk.Entry(adv, textvariable=self.event_cooldown_var).grid(row=2, column=3, sticky="ew", pady=3)

        ttk.Label(adv, text="Ключ шифрования БД (.key)").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=(8, 3))
        ttk.Entry(adv, textvariable=self.key_path_var).grid(
            row=3, column=1, columnspan=3, sticky="ew", pady=(8, 3))

        # ── ночной порог ──────────────────────────────────────────────────────
        nh = ttk.LabelFrame(inner, text="Ночной порог верификации", padding=12)
        nh.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        nh.columnconfigure(1, weight=1)
        nh.columnconfigure(3, weight=1)

        ttk.Checkbutton(nh, text="Строже проверять вне рабочего времени",
                        variable=self.threshold_after_hours_var).grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))
        ttk.Label(nh, text="Порог ночью (0–1)").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(nh, textvariable=self.threshold_night_var).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Label(nh, text="Начало (ч, 0–23)").grid(row=1, column=2, sticky="w", padx=(16, 8), pady=3)
        ttk.Entry(nh, textvariable=self.threshold_night_start_var).grid(row=1, column=3, sticky="ew", pady=3)
        ttk.Label(nh, text="Конец (ч, 0–23)").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(nh, textvariable=self.threshold_night_end_var).grid(row=2, column=1, sticky="ew", pady=3)
        ttk.Label(nh, text="Пример: 20 → 8 = повышенный порог с 20:00 до 08:00",
                  style="Muted.TLabel").grid(row=3, column=0, columnspan=4, sticky="w", pady=(4, 0))

        # ── PIN-защита ────────────────────────────────────────────────────────
        pin_sec = ttk.LabelFrame(inner, text="Защита настроек PIN-кодом", padding=12)
        pin_sec.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Checkbutton(pin_sec, text="Требовать PIN при открытии вкладки «Настройки»",
                        variable=self.pin_enabled_var).pack(anchor="w")
        ttk.Button(pin_sec, text="Установить / изменить PIN",
                   style="Neutral.TButton", command=self._set_pin).pack(
            anchor="w", pady=(8, 0))
        ttk.Label(pin_sec,
                  text="PIN хранится в конфиге в виде SHA-256 хэша.\n"
                       "После установки PIN вкладка будет запрашивать его при каждом переходе.",
                  style="Muted.TLabel").pack(anchor="w", pady=(6, 0))

        ttk.Button(
            inner, text="▶  Запустить верификацию", style="Success.TButton",
            command=self.start_verify,
        ).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 10))

    # ═══════════════════════ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ═══════════════════════════
    def _field(self, parent, row: int, label: str,
               var: tk.Variable, browse: bool = False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w",
                                           padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        if browse:
            ttk.Button(parent, text="Обзор...", style="Neutral.TButton",
                       command=lambda: self._pick_file(var)).grid(
                row=row, column=2, sticky="ew", padx=(8, 0), pady=4)

    def _pick_file(self, var: tk.Variable) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch / SQLite", "*.pt *.db"), ("All files", "*.*")])
        if path:
            var.set(path)

    def _scroll_canvas_event(self, canvas: tk.Canvas, event: tk.Event) -> None:
        if str(event.widget.winfo_class()) in {
                "Text", "Treeview", "Entry", "TEntry", "Combobox", "TCombobox"}:
            return
        delta = int(-event.delta / 120) if event.delta else 0
        if delta:
            canvas.yview_scroll(delta, "units")

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
            "ERROR":    (C_ERROR,   C_WHITE),
            "STOPPING": (C_WARN,    C_WHITE),
            "SETUP":    (C_ACCENT,  C_WHITE),
            "TELEGRAM": ("#0ea5e9", C_WHITE),
            "CAPTURE":  ("#7c3aed", C_WHITE),
            "TRAINING": (C_WARN,    C_WHITE),
        }
        bg, fg = colors.get(mode, ("#374151", "#f9fafb"))
        self._status_pill.configure(bg=bg, fg=fg)

    # ═══════════════════════════ ВАЛИДАЦИЯ ════════════════════════════════════
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
        cmd = [sys.executable, "camera_app.py",
               "--camera-index", str(camera_index),
               "--db-path", str(db_path)]
        if weights_path:
            cmd.extend(["--weights-path", str(weights_path)])
        if device:
            cmd.extend(["--device", device])
        return cmd

    # ═══════════════════════════ ДЕЙСТВИЯ ════════════════════════════════════
    def start_register(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Занято", "Сначала остановите текущий процесс.")
            return
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Ошибка", "Введите имя сотрудника.")
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
            messagebox.showwarning("Занято", "Сначала остановите текущий процесс.")
            return
        try:
            threshold            = float(self.threshold_var.get().strip())
            event_ttl_days       = int(self.event_ttl_days_var.get().strip())
            event_cooldown       = float(self.event_cooldown_var.get().strip())
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
            messagebox.showwarning("Антиспуф-модель не найдена",
                                   "Файл модели не найден.\nВерификация запустится без антиспуфа.")
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
        cmd.append("--anti-spoof" if anti_spoof_enabled else "--no-anti-spoof")
        cmd.extend(["--anti-spoof-threshold", str(anti_spoof_threshold)])
        cmd.extend(["--passive-residual-flow", str(passive_residual)])
        cmd.extend(["--passive-min-texture", str(passive_texture)])
        cmd.extend(["--passive-max-face-jitter-px", str(passive_max_jitter)])
        cmd.append("--relay-enabled" if self.relay_enabled_var.get() else "--no-relay-enabled")
        cmd.extend(["--relay-pin", str(relay_pin)])
        cmd.extend(["--relay-duration-sec", str(relay_duration)])
        cmd.append("--relay-active-high" if self.relay_active_high_var.get()
                   else "--no-relay-active-high")
        cmd.append("--telegram-enabled" if self.telegram_enabled_var.get()
                   else "--no-telegram-enabled")
        if self.telegram_bot_token_var.get().strip():
            cmd.extend(["--telegram-bot-token", self.telegram_bot_token_var.get().strip()])
        if self.telegram_chat_id_var.get().strip():
            cmd.extend(["--telegram-chat-id", self.telegram_chat_id_var.get().strip()])
        if self.telegram_chat_id2_var.get().strip():
            cmd.extend(["--telegram-chat-id2", self.telegram_chat_id2_var.get().strip()])
        cmd.extend(["--telegram-fail-threshold",    str(tg_fail_threshold)])
        cmd.extend(["--telegram-window-sec",         str(tg_window)])
        cmd.extend(["--telegram-alert-cooldown-sec", str(tg_alert_cooldown)])
        cmd.extend(["--telegram-unknown-cooldown-sec", str(tg_unknown_cooldown)])
        cmd.append("--telegram-send-photo" if self.telegram_send_photo_var.get()
                   else "--no-telegram-send-photo")
        cmd.append("--telegram-admin-buttons" if self.telegram_admin_buttons_var.get()
                   else "--no-telegram-admin-buttons")
        cmd.extend(["--telegram-admin-allow-sec", str(tg_admin_allow_sec)])

        if self.threshold_after_hours_var.get():
            cmd.append("--threshold-after-hours")
            cmd.extend(["--threshold-night",       self.threshold_night_var.get().strip()])
            cmd.extend(["--threshold-night-start", self.threshold_night_start_var.get().strip()])
            cmd.extend(["--threshold-night-end",   self.threshold_night_end_var.get().strip()])
        else:
            cmd.append("--no-threshold-after-hours")

        cmd.append("--schedule-enabled" if self.schedule_enabled_var.get()
                   else "--no-schedule-enabled")
        cmd.append("--cluster-unknowns" if self.cluster_unknowns_var.get()
                   else "--no-cluster-unknowns")

        self._start_process(cmd, "VERIFY", "Верификация запущена")
        self.nb.select(0)

    def download_weights(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Занято", "Сначала остановите процесс.")
            return
        self._start_process([sys.executable, "download_weights.py"],
                             "SETUP", "Загрузка весов FaceNet...")

    def test_telegram(self) -> None:
        from src.faceid.alerts import MultiTelegramAlerter, TelegramAlerter as _TA
        token   = self.telegram_bot_token_var.get().strip()
        chat_id = self.telegram_chat_id_var.get().strip()
        chat_id2 = self.telegram_chat_id2_var.get().strip()
        if not token or not chat_id:
            messagebox.showerror("Ошибка", "Укажите Bot Token и Chat ID 1.")
            return
        self.set_status("TELEGRAM", "Отправка тестового сообщения...")
        alerters = [_TA(bot_token=token, chat_id=chat_id)]
        if chat_id2:
            alerters.append(_TA(bot_token=token, chat_id=chat_id2))
        multi = MultiTelegramAlerter(alerters=alerters)
        threading.Thread(target=self._send_test_telegram,
                         args=(multi, len(alerters)), daemon=True).start()

    def _send_test_telegram(self, alerter, count: int) -> None:
        try:
            now = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            alerter.send_html(
                f"&#128994; <b>FaceID — тест уведомлений</b>\n\n"
                f"Подключение к боту успешно.\n"
                f"&#128100; Получателей: <b>{count}</b>\n"
                f"&#128337; <code>{now}</code>"
            )
        except Exception as exc:
            self.after(0, self.append_log, f"Telegram тест: ошибка — {exc}")
            self.after(0, self.set_status, "ERROR", "Telegram ошибка")
            return
        self.after(0, self.append_log, f"Telegram тест: отправлено ({count} чел.)")
        self.after(0, self.set_status, "TELEGRAM", f"Telegram: тест отправлен")

    def start_capture(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Занято", "Сначала остановите текущий процесс.")
            return
        label  = self.capture_label_var.get().strip()
        person = self.capture_person_var.get().strip() or "session1"
        try:
            count  = int(self.capture_count_var.get().strip())
        except ValueError:
            messagebox.showerror("Ошибка", "Кол-во снимков должно быть числом.")
            return
        dataset_dir = self.model_dataset_dir_var.get().strip() or "dataset/anti_spoof"
        try:
            cam = int(self.camera_index_var.get().strip())
        except ValueError:
            cam = 0
        cmd = [
            sys.executable, "anti_spoof_capture.py",
            "--label", label,
            "--person", person,
            "--target-count", str(count),
            "--output-dir", dataset_dir,
            "--camera-index", str(cam),
        ]
        label_ru = "живое лицо" if label == "live" else "фото-подделка"
        self._start_process(cmd, "CAPTURE", f"Захват: {label_ru}")

    def start_train_model(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Занято", "Сначала остановите текущий процесс.")
            return
        dataset_dir = self.model_dataset_dir_var.get().strip() or "dataset/anti_spoof"
        live_dir    = Path(dataset_dir) / "live"
        spoof_dir   = Path(dataset_dir) / "spoof"
        live_cnt    = len(list(live_dir.glob("*.jpg"))) if live_dir.exists() else 0
        spoof_cnt   = len(list(spoof_dir.glob("*.jpg"))) if spoof_dir.exists() else 0
        if live_cnt < 20 or spoof_cnt < 20:
            messagebox.showerror(
                "Недостаточно данных",
                f"Живых снимков: {live_cnt}, подделок: {spoof_cnt}.\n"
                "Нужно минимум 20 снимков каждого типа.")
            return
        if not messagebox.askyesno(
            "Запустить обучение?",
            f"Датасет: {live_cnt} живых / {spoof_cnt} подделок\n\n"
            "Обучение займёт 3–10 минут.\n"
            "Текущая модель будет заменена. Продолжить?"):
            return
        try:
            epochs = int(self.train_epochs_var.get().strip())
        except ValueError:
            epochs = 8
        device = self.device_var.get().strip() or "cpu"
        cmd = [
            sys.executable, "anti_spoof_train.py",
            "--dataset-dir", dataset_dir,
            "--output-model", self.anti_spoof_model_path_var.get().strip()
                              or "artifacts/anti_spoof_model.pt",
            "--epochs", str(epochs),
            "--device", device,
        ]
        self._start_process(cmd, "TRAINING", "Обучение модели антиспуфинга...")

    # ═══════════════════════ УПРАВЛЕНИЕ ПРОЦЕССОМ ══════════════════════════
    def _start_process(self, cmd: list[str], mode: str, status: str) -> None:
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(Path(__file__).parent))
        except Exception as exc:
            self.proc = None
            messagebox.showerror("Ошибка запуска", str(exc))
            return
        self.append_log(">>> " + " ".join(cmd))
        self.set_status(mode, status)
        self.output_thread = threading.Thread(
            target=self._read_output, daemon=True)
        self.output_thread.start()

    def _read_output(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            self.after(0, self.append_log, line.rstrip("\n"))
            self.after(0, self._update_status_from_log, line)
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
        elif "best_val_acc" in tl or "best epoch" in tl.replace(" ", "_"):
            self.after(500, self._refresh_model_tab)

    def _on_process_exit(self, exit_code: int) -> None:
        self.append_log(f"[процесс завершился с кодом {exit_code}]")
        self.proc = None
        if exit_code == 0:
            self.set_status("IDLE", "Ожидание")
            self._refresh_model_tab()
        else:
            self.set_status("ERROR", f"Ошибка (код {exit_code})")

    def stop_process(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        self.append_log("[запрошена остановка]")
        self.set_status("STOPPING", "Остановка...")

    # ═══════════════════════ ОБНОВЛЕНИЕ ДАННЫХ ════════════════════════════════
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
                pass

        self.events_tree.delete(*self.events_tree.get_children())
        for ts_utc, decision, identity, score in rows:
            t_short = str(ts_utc)[11:19] if len(str(ts_utc)) >= 19 else str(ts_utc)
            s_text  = "-" if score is None or float(score) < 0 else f"{float(score):.3f}"
            tag = ("allow" if decision == "ALLOW"
                   else "live" if decision == "DENY_LIVENESS"
                   else "deny" if decision and decision.startswith("DENY")
                   else "other")
            self.events_tree.insert("", "end",
                                    values=(t_short, decision, identity, s_text),
                                    tags=(tag,))
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
                    """SELECT SUM(decision='ALLOW'),
                              SUM(decision LIKE 'DENY%'),
                              SUM(is_suspicious)
                       FROM access_events WHERE ts_utc >= ?""", (today,)
                ).fetchone()
                journal_rows = conn.execute(
                    "SELECT ts_utc, decision, identity, score, reason "
                    "FROM access_events ORDER BY id DESC LIMIT 200"
                ).fetchall()
            finally:
                conn.close()
        except Exception:
            return

        # пользователи (вкладка Сотрудники)
        self.users_tree.delete(*self.users_tree.get_children())
        for name, updated_at in users:
            date = str(updated_at)[:10] if updated_at else "—"
            self.users_tree.insert("", "end", values=(name, date))

        # посещаемость
        attendance = get_attendance_today_utc(Path(db_path))
        self.attend_tree.delete(*self.attend_tree.get_children())
        for name, first_time in attendance:
            self.attend_tree.insert("", "end", values=(name, first_time), tags=("present",))

        # журнал (вкладка Журнал)
        self.journal_tree.delete(*self.journal_tree.get_children())
        for ts_utc, decision, identity, score, reason in journal_rows:
            s_text = "-" if score is None or float(score) < 0 else f"{float(score):.3f}"
            tag = ("allow" if decision == "ALLOW"
                   else "live" if decision == "DENY_LIVENESS"
                   else "deny" if decision and decision.startswith("DENY")
                   else "other")
            self.journal_tree.insert("", "end",
                                     values=(ts_utc, decision, identity, s_text, reason or ""),
                                     tags=(tag,))

        ok, denied, sus = (int(v or 0) for v in (row or (0, 0, 0)))
        self.stats_allow_var.set(str(ok))
        self.stats_deny_var.set(str(denied))
        self.stats_sus_var.set(str(sus))
        self.stats_total_var.set(str(ok + denied))
        self.stats_visitors_var.set(str(len(attendance)))
        self.stats_registered_var.set(str(len(users)))

    def _refresh_model_tab(self) -> None:
        # метрики из JSON
        metrics_path = Path("artifacts/anti_spoof_model.metrics.json")
        if metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                acc   = data.get("best_val_acc", 0)
                lr    = data.get("best_live_recall", 0)
                sr    = data.get("best_spoof_recall", 0)
                ntrain = data.get("num_train", 0)
                nval   = data.get("num_val", 0)
                self.model_acc_var.set(f"{acc*100:.1f}%")
                self.model_live_recall_var.set(f"{lr*100:.1f}%")
                self.model_spoof_recall_var.set(f"{sr*100:.1f}%")
                self.model_dataset_info_var.set(f"{ntrain + nval} фото")
            except Exception:
                pass

        # статистика датасета
        ds_dir = Path(self.model_dataset_dir_var.get().strip() or "dataset/anti_spoof")
        live_cnt  = len(list((ds_dir / "live").glob("*.jpg"))) if (ds_dir / "live").exists() else 0
        spoof_cnt = len(list((ds_dir / "spoof").glob("*.jpg"))) if (ds_dir / "spoof").exists() else 0
        self.dataset_live_var.set(str(live_cnt))
        self.dataset_spoof_var.set(str(spoof_cnt))

    # ═════════════════════════ БАЗА ДАННЫХ (ДЕЙСТВИЯ) ════════════════════════
    def _delete_identity(self) -> None:
        sel = self.users_tree.selection()
        if not sel:
            messagebox.showinfo("Выбор", "Выберите сотрудника для удаления.")
            return
        name = self.users_tree.item(sel[0])["values"][0]
        if not messagebox.askyesno("Удаление",
                                   f"Удалить сотрудника «{name}» из базы?"):
            return
        db_path = self.db_path_var.get().strip()
        try:
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("DELETE FROM identities WHERE name = ?", (name,))
                conn.execute("DELETE FROM identity_embeddings WHERE name = ?", (name,))
                # Remove schedule only if table exists (may not exist on first run)
                tables = {r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                if "identity_schedules" in tables:
                    conn.execute("DELETE FROM identity_schedules WHERE name = ?", (name,))
                conn.commit()
            finally:
                conn.close()
            self.append_log(f"Сотрудник «{name}» удалён из базы данных.")
            self._refresh_db_tab()
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _export_csv(self) -> None:
        from src.faceid.modeling import export_events_csv
        db_raw = self.db_path_var.get().strip()
        if not db_raw or not Path(db_raw).exists():
            messagebox.showerror("Ошибка", "База данных не найдена.")
            return
        out = filedialog.asksaveasfilename(
            title="Экспорт журнала событий", defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
            initialfile="access_events.csv")
        if not out:
            return
        try:
            n = export_events_csv(Path(db_raw), Path(out))
            messagebox.showinfo("Готово", f"Экспортировано {n} записей.")
            self.append_log(f"Экспорт завершён: {n} записей → {out}")
        except Exception as exc:
            messagebox.showerror("Ошибка экспорта", str(exc))

    def _backup_db(self) -> None:
        from src.faceid.modeling import backup_db
        db_raw = self.db_path_var.get().strip()
        if not db_raw or not Path(db_raw).exists():
            messagebox.showerror("Ошибка", "База данных не найдена.")
            return
        out = filedialog.asksaveasfilename(
            title="Резервная копия", defaultextension=".db",
            filetypes=[("SQLite", "*.db"), ("Все файлы", "*.*")],
            initialfile="faceid_backup.db")
        if not out:
            return
        try:
            backup_db(Path(db_raw), Path(out))
            messagebox.showinfo("Готово", "Резервная копия создана.")
            self.append_log(f"Резервная копия: {out}")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _backup_db_encrypted(self) -> None:
        from src.faceid.crypto import backup_encrypted, ensure_key
        db_raw = self.db_path_var.get().strip()
        if not db_raw or not Path(db_raw).exists():
            messagebox.showerror("Ошибка", "База данных не найдена.")
            return
        out = filedialog.asksaveasfilename(
            title="Зашифрованная копия", defaultextension=".enc",
            filetypes=[("Зашифрованные файлы", "*.enc"), ("Все файлы", "*.*")],
            initialfile="faceid_backup.db.enc")
        if not out:
            return
        key_path = Path(self.key_path_var.get().strip() or "artifacts/faceid.key")
        try:
            key = ensure_key(key_path)
            backup_encrypted(Path(db_raw), Path(out), key)
            messagebox.showinfo(
                "Готово",
                f"Зашифрованная копия создана.\n\nКлюч шифрования:\n{key_path}\n\n"
                "Сохраните ключ — без него расшифровка невозможна.")
            self.append_log(f"Зашифрованная копия: {out}")
        except Exception as exc:
            messagebox.showerror("Ошибка шифрования", str(exc))

    # ═══════════════════════ TAB: АНАЛИТИКА ══════════════════════════════════
    def _build_tab_analytics(self) -> None:
        t = self._tab_analytics
        t.columnconfigure(0, weight=3)
        t.columnconfigure(1, weight=2)
        t.rowconfigure(1, weight=2)
        t.rowconfigure(2, weight=1)

        # ── панель кнопок ─────────────────────────────────────────────────────
        btn_row = ttk.Frame(t)
        btn_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Button(btn_row, text="Обновить графики", style="Neutral.TButton",
                   command=self._refresh_analytics).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Проверить цепочку журнала", style="Neutral.TButton",
                   command=self._check_chain_integrity).pack(side=tk.LEFT, padx=(0, 8))
        self._chain_status_var = tk.StringVar(value="")
        ttk.Label(btn_row, textvariable=self._chain_status_var,
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=8)

        # ── тепловая карта ────────────────────────────────────────────────────
        self._heat_frame = ttk.LabelFrame(
            t, text="Активность по часам и дням недели (последние 30 дней)", padding=6)
        self._heat_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        # ── ежедневная посещаемость ───────────────────────────────────────────
        self._att_frame = ttk.LabelFrame(
            t, text="Посещаемость по дням (последние 30 дней)", padding=6)
        self._att_frame.grid(row=1, column=1, sticky="nsew")

        # ── неизвестные посетители ────────────────────────────────────────────
        unk_frame = ttk.LabelFrame(t, text="Неизвестные посетители — кластеризация", padding=8)
        unk_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        unk_frame.rowconfigure(1, weight=1)
        unk_frame.columnconfigure(0, weight=1)

        unk_btn = ttk.Frame(unk_frame)
        unk_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(unk_btn, text="Перекластеризовать", style="Neutral.TButton",
                   command=self._recluster_unknowns).pack(side=tk.LEFT, padx=(0, 8))
        self._cluster_count_var = tk.StringVar(value="")
        ttk.Label(unk_btn, textvariable=self._cluster_count_var,
                  style="Muted.TLabel").pack(side=tk.LEFT)

        ucols = ("id", "visits", "first_seen", "last_seen")
        self.unknown_tree = ttk.Treeview(unk_frame, columns=ucols, show="headings", height=4)
        self.unknown_tree.heading("id",         text="Группа №")
        self.unknown_tree.heading("visits",     text="Визитов")
        self.unknown_tree.heading("first_seen", text="Первый визит")
        self.unknown_tree.heading("last_seen",  text="Последний визит")
        self.unknown_tree.column("id",         width=80,  anchor="center")
        self.unknown_tree.column("visits",     width=80,  anchor="center")
        self.unknown_tree.column("first_seen", width=150, anchor="center")
        self.unknown_tree.column("last_seen",  width=150, anchor="center")
        unk_sb = ttk.Scrollbar(unk_frame, orient=tk.VERTICAL, command=self.unknown_tree.yview)
        self.unknown_tree.configure(yscrollcommand=unk_sb.set)
        self.unknown_tree.grid(row=1, column=0, sticky="nsew")
        unk_sb.grid(row=1, column=1, sticky="ns")

        ttk.Label(unk_frame, text=(
            "Алгоритм DBSCAN группирует неизвестных посетителей по внешности. "
            "Высокое кол-во визитов одной группы — потенциальная угроза безопасности."
        ), style="Muted.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _refresh_analytics(self) -> None:
        db_path = Path(self.db_path_var.get().strip() or "artifacts/faceid_identities.db")
        self._draw_heatmap(self._heat_frame, db_path)
        self._draw_daily_chart(self._att_frame, db_path)
        self._refresh_unknown_clusters(db_path)

    def _draw_heatmap(self, parent: ttk.LabelFrame, db_path: Path) -> None:
        for w in parent.winfo_children():
            w.destroy()
        if not _MPL_OK:
            ttk.Label(parent, text="pip install matplotlib",
                      style="Muted.TLabel").pack(pady=20)
            return
        from src.faceid.modeling import get_hourly_heatmap
        matrix = get_hourly_heatmap(db_path)
        fig = Figure(figsize=(7, 3.2), dpi=90, facecolor=C_WHITE)
        ax = fig.add_subplot(111)
        days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
        im = ax.imshow(matrix, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=7, rotation=45)
        ax.set_yticks(range(7))
        ax.set_yticklabels(days, fontsize=9)
        ax.set_title("Тепловая карта посещений (UTC)", fontsize=10, pad=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout(pad=1.2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _draw_daily_chart(self, parent: ttk.LabelFrame, db_path: Path) -> None:
        for w in parent.winfo_children():
            w.destroy()
        if not _MPL_OK:
            ttk.Label(parent, text="pip install matplotlib",
                      style="Muted.TLabel").pack(pady=20)
            return
        from src.faceid.modeling import get_daily_attendance
        daily = get_daily_attendance(db_path)
        fig = Figure(figsize=(5, 3.2), dpi=90, facecolor=C_WHITE)
        ax = fig.add_subplot(111)
        if daily:
            labels = [d[0][5:] for d in daily]   # MM-DD
            counts = [d[1] for d in daily]
            xs = range(len(labels))
            ax.plot(list(xs), counts, "o-", color="#2563eb", linewidth=2, markersize=5)
            ax.fill_between(list(xs), counts, alpha=0.08, color="#2563eb")
            step = max(1, len(labels) // 8)
            ax.set_xticks(list(xs)[::step])
            ax.set_xticklabels(labels[::step], rotation=45, fontsize=7)
        else:
            ax.text(0.5, 0.5, "Нет данных за последние 30 дней",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="#64748b")
        ax.set_title("Уникальных сотрудников в день", fontsize=10, pad=6)
        ax.set_ylabel("Чел.", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout(pad=1.2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _refresh_unknown_clusters(self, db_path: Path) -> None:
        from src.faceid.modeling import get_unknown_clusters
        clusters = get_unknown_clusters(db_path)
        self.unknown_tree.delete(*self.unknown_tree.get_children())
        for c in clusters:
            self.unknown_tree.insert("", "end", values=(
                f"#{c['cluster_id']}", c["visit_count"],
                c["first_seen"], c["last_seen"],
            ))
        total = sum(c["visit_count"] for c in clusters)
        self._cluster_count_var.set(
            f"{len(clusters)} групп, {total} визитов"
            if clusters else "Нет данных — нажмите «Перекластеризовать»"
        )

    def _recluster_unknowns(self) -> None:
        from src.faceid.modeling import recluster_unknowns
        db_path = Path(self.db_path_var.get().strip() or "artifacts/faceid_identities.db")
        try:
            n = recluster_unknowns(db_path)
            self._refresh_unknown_clusters(db_path)
            self.append_log(f"Кластеризация: найдено {n} групп неизвестных посетителей.")
        except Exception as exc:
            messagebox.showerror("Ошибка кластеризации", str(exc))

    def _check_chain_integrity(self) -> None:
        from src.faceid.modeling import verify_chain_integrity
        db_path = Path(self.db_path_var.get().strip() or "artifacts/faceid_identities.db")
        try:
            ok, count, broken_id = verify_chain_integrity(db_path)
        except Exception as exc:
            self._chain_status_var.set(f"Ошибка: {exc}")
            return
        if count == 0:
            self._chain_status_var.set("Журнал пуст")
            return
        if ok:
            self._chain_status_var.set(f"Цепочка цела ({count} записей)")
            # green label
            for w in self.nb.nametowidget(self.nb.tabs()[5]).winfo_children():
                pass  # no easy way to recolor ttk label, just set text
        else:
            self._chain_status_var.set(f"НАРУШЕНИЕ цепочки! Запись #{broken_id}")
            messagebox.showwarning(
                "Нарушение целостности журнала",
                f"Обнаружено нарушение хэш-цепочки в записи #{broken_id}.\n"
                "Возможна несанкционированная модификация журнала доступа.")

    # ─────────────────────────────────────────────────────────────────────────
    def _show_schedule_dialog(self) -> None:
        from src.faceid.modeling import get_schedule, upsert_schedule
        sel = self.users_tree.selection()
        if not sel:
            messagebox.showinfo("Выбор", "Выберите сотрудника в списке.")
            return
        name = self.users_tree.item(sel[0])["values"][0]
        db_path = Path(self.db_path_var.get().strip())
        schedule = get_schedule(db_path, name) if db_path.exists() else []
        sched_by_day = {e["weekday"]: e for e in schedule}

        dlg = tk.Toplevel(self)
        dlg.title(f"Расписание доступа: {name}")
        dlg.geometry("450x380")
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()
        dlg.configure(bg=C_BG)

        frame = ttk.Frame(dlg, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text=f"Разрешённые часы входа — {name}",
                  style="Section.TLabel").grid(row=0, column=0, columnspan=4,
                                               sticky="w", pady=(0, 10))
        for col, txt in enumerate(["День", "Активно", "С (ЧЧ:ММ)", "По (ЧЧ:ММ)"]):
            ttk.Label(frame, text=txt, font=("Segoe UI", 9, "bold")).grid(
                row=1, column=col, sticky="w", padx=(0 if col == 0 else 8, 0), pady=(0, 4))

        day_names = ["Понедельник", "Вторник", "Среда", "Четверг",
                     "Пятница", "Суббота", "Воскресенье"]
        enabled_vars, start_vars, end_vars = [], [], []
        for i, day_name in enumerate(day_names):
            e = sched_by_day.get(i)
            ev = tk.BooleanVar(value=e is not None)
            sv = tk.StringVar(value=f"{e['start_hour']:02d}:{e.get('start_min', 0):02d}"
                              if e else "08:00")
            nv = tk.StringVar(value=f"{e['end_hour']:02d}:{e.get('end_min', 0):02d}"
                              if e else "18:00")
            ttk.Label(frame, text=day_name).grid(row=i + 2, column=0, sticky="w", pady=3)
            ttk.Checkbutton(frame, variable=ev).grid(row=i + 2, column=1, padx=8, pady=3)
            ttk.Entry(frame, textvariable=sv, width=8).grid(row=i + 2, column=2, padx=8, pady=3)
            ttk.Entry(frame, textvariable=nv, width=8).grid(row=i + 2, column=3, padx=8, pady=3)
            enabled_vars.append(ev)
            start_vars.append(sv)
            end_vars.append(nv)

        def _save():
            entries = []
            for i in range(7):
                if not enabled_vars[i].get():
                    continue
                try:
                    sh, sm = (int(x) for x in start_vars[i].get().split(":"))
                    eh, em = (int(x) for x in end_vars[i].get().split(":"))
                except ValueError:
                    messagebox.showerror("Ошибка",
                                         f"Неверный формат времени в строке {i + 1}.")
                    return
                entries.append({"weekday": i, "start_hour": sh, "start_min": sm,
                                 "end_hour": eh, "end_min": em})
            try:
                upsert_schedule(db_path, name, entries)
                self.append_log(f"Расписание сохранено для «{name}» ({len(entries)} дней).")
                dlg.destroy()
            except Exception as exc:
                messagebox.showerror("Ошибка", str(exc))

        btn_row = ttk.Frame(dlg, padding=(16, 0, 16, 14))
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Сохранить", style="Success.TButton",
                   command=_save).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Отмена", style="Neutral.TButton",
                   command=dlg.destroy).pack(side=tk.LEFT)

    def _export_pdf(self) -> None:
        from src.faceid.modeling import export_pdf_report
        db_raw = self.db_path_var.get().strip()
        if not db_raw or not Path(db_raw).exists():
            messagebox.showerror("Ошибка", "База данных не найдена.")
            return
        start = self.report_start_var.get().strip()
        end   = self.report_end_var.get().strip()
        out = filedialog.asksaveasfilename(
            title="Экспорт PDF-отчёта", defaultextension=".pdf",
            filetypes=[("PDF файлы", "*.pdf"), ("Все файлы", "*.*")],
            initialfile=f"faceid_report_{start}_{end}.pdf")
        if not out:
            return
        try:
            n = export_pdf_report(Path(db_raw), start, end, Path(out))
            messagebox.showinfo("Готово", f"PDF-отчёт сформирован: {n} строк.")
            self.append_log(f"PDF-отчёт: {n} строк → {out}")
        except Exception as exc:
            messagebox.showerror("Ошибка PDF", str(exc))

    def _send_pdf_to_telegram(self) -> None:
        token   = self.telegram_bot_token_var.get().strip()
        chat_id = self.telegram_chat_id_var.get().strip()
        if not token or not chat_id:
            messagebox.showerror(
                "Telegram не настроен",
                "Укажите Bot Token и Chat ID во вкладке Настройки → Telegram.")
            return
        db_raw = self.db_path_var.get().strip()
        if not db_raw or not Path(db_raw).exists():
            messagebox.showerror("Ошибка", "База данных не найдена.")
            return
        start = self.report_start_var.get().strip()
        end   = self.report_end_var.get().strip()
        self.set_status("TELEGRAM", f"Формирование PDF {start}…{end}")
        threading.Thread(
            target=self._send_pdf_worker,
            args=(token, chat_id, self.telegram_chat_id2_var.get().strip(),
                  Path(db_raw), start, end),
            daemon=True,
        ).start()

    def _send_pdf_worker(self, token: str, chat_id: str, chat_id2: str,
                          db_path: Path, start: str, end: str) -> None:
        import tempfile
        from src.faceid.modeling import export_pdf_report
        from src.faceid.alerts import TelegramAlerter, MultiTelegramAlerter
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            n = export_pdf_report(db_path, start, end, tmp_path)
            pdf_bytes = tmp_path.read_bytes()
            tmp_path.unlink(missing_ok=True)

            alerters = [TelegramAlerter(bot_token=token, chat_id=chat_id)]
            if chat_id2:
                alerters.append(TelegramAlerter(bot_token=token, chat_id=chat_id2))
            multi = MultiTelegramAlerter(alerters=alerters)

            filename = f"faceid_report_{start}_{end}.pdf"
            caption = (
                f"📄 FaceID — Отчёт о посещаемости\n"
                f"Период: {start} — {end}\n"
                f"Строк: {n}"
            )
            multi.send_document(pdf_bytes, filename, caption=caption)

            self.after(0, self.append_log,
                       f"PDF отправлен в Telegram ({len(alerters)} получ.): {n} строк.")
            self.after(0, self.set_status, "TELEGRAM", "PDF отправлен")
        except Exception as exc:
            self.after(0, self.append_log, f"Ошибка отправки PDF: {exc}")
            self.after(0, self.set_status, "ERROR", "Ошибка отправки PDF")

    # ═══════════════════════ АВТО-ОБНОВЛЕНИЕ / ФОТО / PIN ════════════════════
    def _auto_refresh_stats(self) -> None:
        self._refresh_db_tab()
        ts = datetime.datetime.now().strftime("обновлено %H:%M:%S")
        self.last_refresh_var.set(ts)
        self.after(5000, self._auto_refresh_stats)

    def _on_user_select(self, _event=None) -> None:
        sel = self.users_tree.selection()
        if not sel:
            return
        name = self.users_tree.item(sel[0])["values"][0]
        photo_file = Path("artifacts/photos") / f"{name}.jpg"
        if not _PIL_OK or not photo_file.exists():
            self._photo_label.config(image="", text="", width=0)
            return
        try:
            img = Image.open(photo_file)
            img.thumbnail((120, 120))
            tk_img = ImageTk.PhotoImage(img)
            self._photo_ref = tk_img
            self._photo_label.config(image=tk_img, text="")
        except Exception:
            self._photo_label.config(image="", text="")

    def _export_attendance_report(self) -> None:
        from src.faceid.modeling import export_attendance_report
        db_raw = self.db_path_var.get().strip()
        if not db_raw or not Path(db_raw).exists():
            messagebox.showerror("Ошибка", "База данных не найдена.")
            return
        start = self.report_start_var.get().strip()
        end   = self.report_end_var.get().strip()
        out = filedialog.asksaveasfilename(
            title="Отчёт о посещаемости", defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
            initialfile=f"attendance_{start}_{end}.csv")
        if not out:
            return
        try:
            n = export_attendance_report(Path(db_raw), start, end, Path(out))
            messagebox.showinfo("Готово", f"Экспортировано {n} строк за период {start} — {end}.")
            self.append_log(f"Отчёт посещаемости: {n} строк → {out}")
        except Exception as exc:
            messagebox.showerror("Ошибка экспорта", str(exc))

    def _on_tab_change(self, _event=None) -> None:
        if not self.pin_enabled_var.get():
            return
        pin_hash = self.pin_hash_var.get()
        if not pin_hash:
            return
        # Settings tab is index 4
        try:
            selected = self.nb.index(self.nb.select())
        except Exception:
            return
        if selected != 4:
            return
        self._pin_attempts = 0
        self._ask_pin()

    def _ask_pin(self) -> None:
        entered = simpledialog.askstring(
            "PIN-код", "Введите PIN для доступа к настройкам:",
            show="*", parent=self)
        if entered is None:
            self.nb.select(0)
            return
        pin_hash = hashlib.sha256(entered.encode()).hexdigest()
        if pin_hash == self.pin_hash_var.get():
            return  # access granted
        self._pin_attempts = getattr(self, "_pin_attempts", 0) + 1
        if self._pin_attempts >= 3:
            messagebox.showerror("Отказано", "Слишком много попыток. Доступ заблокирован.")
            self.nb.select(0)
        else:
            messagebox.showerror("Ошибка", f"Неверный PIN. Попытка {self._pin_attempts}/3.")
            self._ask_pin()

    def _set_pin(self) -> None:
        new_pin = simpledialog.askstring(
            "Установить PIN", "Введите новый PIN-код (минимум 4 символа):",
            show="*", parent=self)
        if not new_pin:
            return
        if len(new_pin) < 4:
            messagebox.showerror("Ошибка", "PIN должен содержать не менее 4 символов.")
            return
        confirm = simpledialog.askstring(
            "Подтверждение PIN", "Повторите PIN:", show="*", parent=self)
        if confirm != new_pin:
            messagebox.showerror("Ошибка", "PIN-коды не совпадают.")
            return
        self.pin_hash_var.set(hashlib.sha256(new_pin.encode()).hexdigest())
        self.pin_enabled_var.set(True)
        messagebox.showinfo("Готово", "PIN-код установлен. Защита активирована.")

    # ════════════════════════ КОНФИГ / ЗАКРЫТИЕ ═══════════════════════════════
    _CONFIG_PATH = Path("artifacts/config.json")

    _CONFIG_VARS: list[str] = [
        "camera_index_var", "db_path_var", "device_var", "weights_path_var",
        "anti_spoof_model_path_var",
        "samples_var", "threshold_var",
        "liveness_enabled_var", "show_score_var",
        "event_ttl_days_var", "event_cooldown_var",
        "passive_residual_var", "passive_texture_var", "passive_max_jitter_var",
        "anti_spoof_enabled_var", "anti_spoof_threshold_var",
        "relay_enabled_var", "relay_pin_var", "relay_duration_var",
        "relay_active_high_var",
        "telegram_enabled_var", "telegram_bot_token_var",
        "telegram_chat_id_var", "telegram_chat_id2_var",
        "telegram_fail_threshold_var", "telegram_window_var",
        "telegram_alert_cooldown_var", "telegram_unknown_cooldown_var",
        "telegram_send_photo_var", "telegram_admin_buttons_var",
        "telegram_admin_allow_sec_var",
        "key_path_var",
        "capture_label_var", "capture_person_var", "capture_count_var",
        "model_dataset_dir_var", "train_epochs_var",
        "pin_enabled_var", "pin_hash_var",
        "threshold_after_hours_var", "threshold_night_var",
        "threshold_night_start_var", "threshold_night_end_var",
        "schedule_enabled_var", "cluster_unknowns_var",
    ]

    def _load_config(self) -> None:
        try:
            data = json.loads(self._CONFIG_PATH.read_text(encoding="utf-8"))
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
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _on_close(self) -> None:
        self._save_config()
        if self.proc is not None:
            self.proc.terminate()
        self.destroy()


def main() -> None:
    app = FaceIdGui()
    app.mainloop()


if __name__ == "__main__":
    main()
