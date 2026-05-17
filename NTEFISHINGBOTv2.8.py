import cv2
import numpy as np
import mss
import pydirectinput
import time
import keyboard
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import threading
import json
import os
from PIL import Image, ImageTk

# ─── ДЕФОЛТНЫЕ НАСТРОЙКИ ──────────────────────────────────────────────────────
CONFIG_FILE = "bot_config.json"

DEFAULTS = {
    "hotkey":             "F8",
    "top":                50,
    "left":               610,
    "width":              710,
    "height":             40,
    "deadzone":           15,
    "auto_click_delay":   5,
    "yellow_h_min":       20,
    "yellow_h_max":       40,
    "yellow_s_min":       150,
    "yellow_v_min":       150,
    "safe_h_min":         80,
    "safe_h_max":         100,
    "safe_s_min":         150,
    "safe_v_min":         150,
    "ui_bg":              "#1e1e2e",
    "ui_fg":              "#cdd6f4",
    "ui_button_bg":       "#89b4fa",
    "ui_button_fg":       "#1e1e2e",
    "ui_accent":          "#f38ba8",
    "ui_status_bg":       "#313244",
    # ── AI ──
    "ai_mode":            True,
    "pid_kp":             0.8,
    "pid_ki":             0.05,
    "pid_kd":             0.30,
    "kalman_proc_noise":  2.0,
    "kalman_meas_noise":  8.0,
    "predict_frames":     2,
    # ── Умный счётчик ──
    "min_success_ratio":  0.30,   # минимум % кадров в зоне чтобы засчитать рыбу
    "min_minigame_secs":  2.0,    # минимальная длительность мини-игры (сек)
}

pydirectinput.PAUSE = 0


# ═══════════════════════════════════════════════════════════════════════════════
#  AI КЛАССЫ
# ═══════════════════════════════════════════════════════════════════════════════

class KalmanFilter1D:
    """
    Одномерный фильтр Калмана — вектор состояния [позиция, скорость].

    Зачем:
      • Убирает шум HSV-детектора (ложные срабатывания, дрожание)
      • Через predict_ahead() даёт предсказание позиции через N кадров,
        чтобы нажать клавишу ДО того, как курсор вышел из зоны.
    """

    def __init__(self, process_noise: float = 2.0, measurement_noise: float = 8.0):
        # Вектор состояния: [position, velocity]
        self.x = np.array([0.0, 0.0])
        # Матрица ошибки оценки
        self.P = np.eye(2) * 100.0
        # Матрица перехода состояния (dt = 1 кадр)
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        # Матрица наблюдения (измеряем только позицию)
        self.H = np.array([[1.0, 0.0]])
        # Шум процесса (как быстро меняется состояние)
        self.Q = np.diag([process_noise, process_noise * 0.5])
        # Шум измерений (погрешность HSV-детектора)
        self.R = np.array([[measurement_noise]])
        self.initialized = False

    def reset(self):
        self.x = np.array([0.0, 0.0])
        self.P = np.eye(2) * 100.0
        self.initialized = False

    def update(self, measurement: float) -> float:
        """Принять новое измерение, вернуть отфильтрованную позицию."""
        if not self.initialized:
            self.x[0] = float(measurement)
            self.initialized = True
            return float(measurement)

        # ── Шаг предсказания ──────────────────────────────
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # ── Шаг обновления ────────────────────────────────
        z   = np.array([float(measurement)])
        y   = z - self.H @ self.x                          # инновация
        S   = self.H @ self.P @ self.H.T + self.R          # ковариация инновации
        K   = (self.P @ self.H.T) / float(S[0, 0])         # коэффициент Калмана
        self.x = self.x + K.reshape(2) * float(y[0])
        self.P = (np.eye(2) - np.outer(K, self.H)) @ self.P

        return float(self.x[0])

    def predict_ahead(self, steps: int = 1) -> float:
        """Предсказать позицию через N кадров (линейная экстраполяция)."""
        return float(self.x[0] + self.x[1] * steps)

    @property
    def velocity(self) -> float:
        return float(self.x[1])

    @property
    def position(self) -> float:
        return float(self.x[0])


class PIDController:
    """
    ПИД-регулятор для плавного управления курсором.

    Заменяет грубый bang-bang (просто держать A или D) на вычисленное
    оптимальное усилие:
      • P — реагирует на текущую ошибку
      • I — убирает систематическое смещение
      • D — демпфирует колебания

    Выход нормирован в [-1, 1]:
      > 0  →  курсор правее зоны  →  жать A
      < 0  →  курсор левее зоны   →  жать D
    """

    def __init__(self, kp=0.8, ki=0.05, kd=0.30, output_limits=(-1.0, 1.0)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limits = output_limits
        self._integral   = 0.0
        self._prev_error = 0.0
        self._ready      = False

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._ready      = False

    def compute(self, error: float, dt: float = 0.05) -> float:
        if dt <= 0:
            dt = 0.05

        # P
        p = self.kp * error

        # I  — anti-windup через клиппинг накопителя
        self._integral += error * dt
        max_i = 50.0 / max(self.ki, 1e-6)
        self._integral = float(np.clip(self._integral, -max_i, max_i))
        i = self.ki * self._integral

        # D
        d = (self.kd * (error - self._prev_error) / dt) if self._ready else 0.0
        self._ready      = True
        self._prev_error = error

        output = p + i + d
        return float(np.clip(output, self.limits[0], self.limits[1]))


# ═══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            for k, v in DEFAULTS.items():
                data.setdefault(k, v)
            return data
    except Exception:
        pass
    return dict(DEFAULTS)


def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)


def get_positions(sct, region, cfg):
    """HSV-детектор позиций курсора (жёлтый) и безопасной зоны (зелёный)."""
    img_bgra = np.array(sct.grab(region))
    img_bgr  = img_bgra[:, :, :3]
    hsv_img  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    yellow_lower = np.array([cfg["yellow_h_min"], cfg["yellow_s_min"], cfg["yellow_v_min"]])
    yellow_upper = np.array([cfg["yellow_h_max"], 255, 255])
    safe_lower   = np.array([cfg["safe_h_min"],   cfg["safe_s_min"],   cfg["safe_v_min"]])
    safe_upper   = np.array([cfg["safe_h_max"],   255,                 255])

    mask_yellow = cv2.inRange(hsv_img, yellow_lower, yellow_upper)
    mask_safe   = cv2.inRange(hsv_img, safe_lower,   safe_upper)

    MIN_PIXEL_AREA = 50

    cursor_x  = None
    M_yellow  = cv2.moments(mask_yellow)
    if M_yellow["m00"] > MIN_PIXEL_AREA:
        cursor_x = int(M_yellow["m10"] / M_yellow["m00"])

    safezone_x = None
    M_safe     = cv2.moments(mask_safe)
    if M_safe["m00"] > MIN_PIXEL_AREA:
        safezone_x = int(M_safe["m10"] / M_safe["m00"])

    return cursor_x, safezone_x


def release_keys():
    pydirectinput.keyUp('a')
    pydirectinput.keyUp('d')


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════════

class FishingBotGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NTE Fisher v2.8  ·  AI + Smart Counter")
        self.root.geometry("480x760")
        self.root.resizable(False, False)
        self.root.attributes('-topmost', True)

        self.cfg          = load_config()
        self.bot_running  = False
        self.bot_thread   = None
        self._hotkey_wait = False
        self.fish_caught  = 0
        self.fish_missed  = 0

        # ── Live AI телеметрия (обновляется из бот-потока) ──
        self._tele_cursor_pos = tk.StringVar(value="—")
        self._tele_cursor_vel = tk.StringVar(value="—")
        self._tele_safe_pos   = tk.StringVar(value="—")
        self._tele_error      = tk.StringVar(value="—")
        self._tele_pid_out    = tk.StringVar(value="—")
        self._tele_confidence = tk.StringVar(value="—")
        self._last_result_var = tk.StringVar(value="—")

        self._init_vars()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')

        self.main_tab     = tk.Frame(self.notebook)
        self.settings_tab = tk.Frame(self.notebook)
        self.ai_tab       = tk.Frame(self.notebook)
        self.ui_tab       = tk.Frame(self.notebook)
        self.stats_tab    = tk.Frame(self.notebook)

        self.notebook.add(self.main_tab,     text="  Main  ")
        self.notebook.add(self.settings_tab, text=" Settings ")
        self.notebook.add(self.ai_tab,       text="  🤖 AI  ")
        self.notebook.add(self.ui_tab,       text="  UI Skin ")
        self.notebook.add(self.stats_tab,    text="  Stats  ")

        self._build_main_tab()
        self._build_settings_tab()
        self._build_ai_tab()
        self._build_ui_tab()
        self._build_stats_tab()

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.sct = mss.MSS()
        self.root.after(150, self._preview_loop)
        self._register_hotkey(self.cfg["hotkey"])
        self._apply_ui_colors()
        self._update_ai_badge()

    # ──────────────────────────────────────────────────────────────────────────
    #  VARS
    # ──────────────────────────────────────────────────────────────────────────

    def _init_vars(self):
        c = self.cfg
        self.var_hotkey   = tk.StringVar(value=c["hotkey"])
        self.var_top      = tk.IntVar(value=c["top"])
        self.var_left     = tk.IntVar(value=c["left"])
        self.var_width    = tk.IntVar(value=c["width"])
        self.var_height   = tk.IntVar(value=c["height"])
        self.var_deadzone = tk.IntVar(value=c["deadzone"])
        self.var_ac_delay = tk.DoubleVar(value=c["auto_click_delay"])

        self.var_ui_bg        = tk.StringVar(value=c["ui_bg"])
        self.var_ui_fg        = tk.StringVar(value=c["ui_fg"])
        self.var_ui_button_bg = tk.StringVar(value=c["ui_button_bg"])
        self.var_ui_button_fg = tk.StringVar(value=c["ui_button_fg"])
        self.var_ui_accent    = tk.StringVar(value=c["ui_accent"])
        self.var_ui_status_bg = tk.StringVar(value=c["ui_status_bg"])

        # AI vars
        self.var_ai_mode        = tk.BooleanVar(value=c.get("ai_mode", True))
        self.var_pid_kp         = tk.DoubleVar(value=c.get("pid_kp", 0.8))
        self.var_pid_ki         = tk.DoubleVar(value=c.get("pid_ki", 0.05))
        self.var_pid_kd         = tk.DoubleVar(value=c.get("pid_kd", 0.30))
        self.var_kalman_proc    = tk.DoubleVar(value=c.get("kalman_proc_noise", 2.0))
        self.var_kalman_meas    = tk.DoubleVar(value=c.get("kalman_meas_noise", 8.0))
        self.var_predict_frames   = tk.IntVar(value=c.get("predict_frames", 2))
        self.var_min_success_ratio = tk.DoubleVar(value=c.get("min_success_ratio", 0.30))
        self.var_min_minigame_secs = tk.DoubleVar(value=c.get("min_minigame_secs", 2.0))

    # ──────────────────────────────────────────────────────────────────────────
    #  TAB: MAIN
    # ──────────────────────────────────────────────────────────────────────────

    def _build_main_tab(self):
        tk.Label(self.main_tab, text="NTE Fisher", font=("Helvetica", 18, "bold")).pack(pady=(20, 2))
        tk.Label(self.main_tab, text="v2.7  ·  AI Edition", font=("Helvetica", 9), fg="#888").pack()

        self.status_label = tk.Label(self.main_tab, text="● IDLE", font=("Helvetica", 13, "bold"))
        self.status_label.pack(pady=(10, 2))

        self.ai_badge = tk.Label(self.main_tab, text="", font=("Helvetica", 9))
        self.ai_badge.pack(pady=(0, 8))

        btn_cfg = dict(font=("Helvetica", 12, "bold"), relief=tk.FLAT,
                       padx=20, pady=8, cursor="hand2", takefocus=0)

        self.start_btn = tk.Button(self.main_tab, text="▶  START BOT",
                                    command=self.start_bot, **btn_cfg)
        self.start_btn.pack(fill=tk.X, padx=50, pady=4)

        self.stop_btn = tk.Button(self.main_tab, text="■  STOP BOT",
                                   command=self.stop_bot, state=tk.DISABLED, **btn_cfg)
        self.stop_btn.pack(fill=tk.X, padx=50, pady=4)

        ttk.Separator(self.main_tab, orient='horizontal').pack(fill=tk.X, padx=20, pady=15)

        tk.Button(self.main_tab, text="💾  Сохранить все настройки",
                  command=self._save_all, **btn_cfg).pack(fill=tk.X, padx=50, pady=4)
        tk.Button(self.main_tab, text="↺  Сбросить к умолчаниям",
                  command=self._reset_defaults, **btn_cfg).pack(fill=tk.X, padx=50, pady=4)

    # ──────────────────────────────────────────────────────────────────────────
    #  TAB: AI
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ai_tab(self):
        # ── Переключатель режима ──────────────────────────────────────────────
        tf = tk.LabelFrame(self.ai_tab, text=" AI Режим ",
                           font=("Helvetica", 9, "bold"), padx=8, pady=6)
        tf.pack(fill=tk.X, padx=10, pady=8)

        tk.Checkbutton(tf, text="Включить AI  (Kalman Filter + PID)",
                       variable=self.var_ai_mode, font=("Helvetica", 10, "bold"),
                       command=self._update_ai_badge).pack(anchor="w")
        tk.Label(tf,
                 text="AI фильтрует шум HSV-детектора и предсказывает\n"
                      "положение курсора вперёд — реакция упреждающая,\n"
                      "а не запаздывающая. PID даёт плавное управление.",
                 font=("Helvetica", 8), justify=tk.LEFT, fg="#888").pack(anchor="w", pady=(4, 0))

        # ── Фильтр Калмана ────────────────────────────────────────────────────
        kf = tk.LabelFrame(self.ai_tab, text=" Фильтр Калмана ",
                           font=("Helvetica", 9, "bold"), padx=6, pady=4)
        kf.pack(fill=tk.X, padx=10, pady=4)

        for label, var, lo, hi, res in [
            ("Шум процесса  (Q)",      self.var_kalman_proc,    0.1, 20.0, 0.1),
            ("Шум измерений (R)",      self.var_kalman_meas,    1.0, 50.0, 0.5),
            ("Предсказание (кадры)",   self.var_predict_frames, 0,    5,   1  ),
        ]:
            row = tk.Frame(kf); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, width=22, anchor="w",
                     font=("Helvetica", 8)).pack(side=tk.LEFT)
            tk.Scale(row, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     variable=var, length=185).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, width=5, anchor="w",
                     font=("Helvetica", 8)).pack(side=tk.LEFT)

        tk.Label(kf,
                 text="Q высокое → доверяем измерениям больше (быстрее, нестабильнее)\n"
                      "R высокое → доверяем фильтру больше (плавнее, но медленнее)",
                 font=("Helvetica", 7), fg="#888", justify=tk.LEFT).pack(anchor="w")

        # ── Умный счётчик ─────────────────────────────────────────────────────
        cf = tk.LabelFrame(self.ai_tab, text=" 🐟 Умный Счётчик ",
                           font=("Helvetica", 9, "bold"), padx=6, pady=4)
        cf.pack(fill=tk.X, padx=10, pady=4)

        for label, var, lo, hi, res in [
            ("Мин. % попаданий в зону", self.var_min_success_ratio, 0.0, 1.0, 0.05),
            ("Мин. длит. мини-игры (с)", self.var_min_minigame_secs, 0.5, 10.0, 0.5),
        ]:
            row = tk.Frame(cf); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, width=25, anchor="w",
                     font=("Helvetica", 8)).pack(side=tk.LEFT)
            tk.Scale(row, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     variable=var, length=160).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, width=5, anchor="w",
                     font=("Helvetica", 8)).pack(side=tk.LEFT)

        tk.Label(cf,
                 text="Рыба засчитывается ТОЛЬКО если оба условия выполнены.\n"
                      "Слишком высокий % = меньше засчитает, слишком низкий = будут ложные срабатывания.",
                 font=("Helvetica", 7), fg="#888", justify=tk.LEFT).pack(anchor="w")

        # ── PID ───────────────────────────────────────────────────────────────
        pf = tk.LabelFrame(self.ai_tab, text=" PID Регулятор ",
                           font=("Helvetica", 9, "bold"), padx=6, pady=4)
        pf.pack(fill=tk.X, padx=10, pady=4)

        for label, var, lo, hi, res in [
            ("kP  пропорциональный",  self.var_pid_kp, 0.0, 3.0, 0.05),
            ("kI  интегральный",      self.var_pid_ki, 0.0, 0.5, 0.01),
            ("kD  дифференциальный",  self.var_pid_kd, 0.0, 2.0, 0.05),
        ]:
            row = tk.Frame(pf); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, width=22, anchor="w",
                     font=("Helvetica", 8)).pack(side=tk.LEFT)
            tk.Scale(row, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     variable=var, length=185).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, width=5, anchor="w",
                     font=("Helvetica", 8)).pack(side=tk.LEFT)

        tk.Label(pf,
                 text="kP → основная реакция  |  kI → убирает drift  |  kD → демпфирует",
                 font=("Helvetica", 7), fg="#888").pack(anchor="w")

        # ── Live телеметрия ───────────────────────────────────────────────────
        lf = tk.LabelFrame(self.ai_tab, text=" Live Телеметрия (обновляется во время рыбалки) ",
                           font=("Helvetica", 9, "bold"), padx=8, pady=4)
        lf.pack(fill=tk.X, padx=10, pady=6)

        tele = [
            ("Курсор (Kalman):",      self._tele_cursor_pos),
            ("Скорость курсора:",     self._tele_cursor_vel),
            ("Safezone (Kalman):",    self._tele_safe_pos),
            ("Ошибка (error):",       self._tele_error),
            ("PID выход:",            self._tele_pid_out),
            ("Confidence детектора:", self._tele_confidence),
        ]
        for lbl, var in tele:
            row = tk.Frame(lf); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=lbl, width=24, anchor="w",
                     font=("Helvetica", 8)).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var,
                     font=("Courier", 9, "bold"), fg="#a6e3a1").pack(side=tk.LEFT)

    # ──────────────────────────────────────────────────────────────────────────
    #  TAB: SETTINGS
    # ──────────────────────────────────────────────────────────────────────────

    def _build_settings_tab(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        rf = tk.LabelFrame(self.settings_tab, text=" Область захвата ",
                           font=("Helvetica", 9, "bold"), padx=6, pady=4)
        rf.pack(fill=tk.X, padx=10, pady=6)

        for label, var, lo, hi in [
            ("Top",    self.var_top,    1, sh),
            ("Left",   self.var_left,   1, sw),
            ("Width",  self.var_width,  10, sw),
            ("Height", self.var_height, 5, 500),
        ]:
            row = tk.Frame(rf); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, width=18, anchor="w").pack(side=tk.LEFT)
            tk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL,
                     variable=var, length=220).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, width=5, anchor="w").pack(side=tk.LEFT)

        bf = tk.LabelFrame(self.settings_tab, text=" Параметры бота ",
                           font=("Helvetica", 9, "bold"), padx=6, pady=4)
        bf.pack(fill=tk.X, padx=10, pady=6)

        for label, var, lo, hi, res in [
            ("Deadzone (px)",        self.var_deadzone, 0,    100, 1  ),
            ("Auto-click delay (s)", self.var_ac_delay, 1.0,  30, 0.5),
        ]:
            row = tk.Frame(bf); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, width=18, anchor="w").pack(side=tk.LEFT)
            tk.Scale(row, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     variable=var, length=220).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, width=5, anchor="w").pack(side=tk.LEFT)

        hf = tk.LabelFrame(self.settings_tab, text=" Горячая клавиша ",
                           font=("Helvetica", 9, "bold"), padx=6, pady=4)
        hf.pack(fill=tk.X, padx=10, pady=6)

        row = tk.Frame(hf); row.pack(pady=4)
        tk.Label(row, text="Старт / Стоп:").pack(side=tk.LEFT, padx=5)
        self.hotkey_entry = tk.Entry(row, textvariable=self.var_hotkey,
                                      width=8, state="readonly",
                                      relief=tk.SUNKEN, takefocus=0)
        self.hotkey_entry.pack(side=tk.LEFT, padx=5)
        self.hotkey_btn = tk.Button(row, text="Сменить",
                                     command=self._begin_hotkey_capture, takefocus=0)
        self.hotkey_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(self.settings_tab, text="Live Preview:",
                 font=("Helvetica", 9, "bold")).pack(pady=(4, 0))
        self.preview_canvas = tk.Label(self.settings_tab, bg="black",
                                        text="Loading…", width=360, height=80)
        self.preview_canvas.pack(padx=10, pady=4, fill=tk.X)

    # ──────────────────────────────────────────────────────────────────────────
    #  TAB: UI SKIN
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui_tab(self):
        info = tk.Label(self.ui_tab,
                        text="Настройка цветов интерфейса.\nИзменения применяются сразу.",
                        font=("Helvetica", 8), justify=tk.LEFT)
        info.pack(padx=10, pady=(8, 2), anchor="w")

        colors = [
            ("Фон окна:",       self.var_ui_bg,        self._apply_ui_colors),
            ("Цвет текста:",    self.var_ui_fg,        self._apply_ui_colors),
            ("Фон кнопок:",     self.var_ui_button_bg, self._apply_ui_colors),
            ("Текст кнопок:",   self.var_ui_button_fg, self._apply_ui_colors),
            ("Акцентный цвет:", self.var_ui_accent,    self._apply_ui_colors),
        ]
        for label, var, callback in colors:
            frame = tk.Frame(self.ui_tab); frame.pack(fill=tk.X, padx=10, pady=3)
            tk.Label(frame, text=label, width=18, anchor="w").pack(side=tk.LEFT)
            color_preview = tk.Label(frame, bg=var.get(), width=4, height=1, relief=tk.SUNKEN)
            color_preview.pack(side=tk.LEFT, padx=5)
            def make_cb(v=var, cp=color_preview, cb=callback):
                def _pick():
                    color = colorchooser.askcolor(initialcolor=v.get())[1]
                    if color:
                        v.set(color); cp.configure(bg=color); cb()
                return _pick
            tk.Button(frame, text="Выбрать", command=make_cb(),
                      bg="#89b4fa", fg="#1e1e2e", takefocus=0).pack(side=tk.LEFT, padx=5)

        tk.Button(self.ui_tab, text="🎨 Сбросить цвета",
                  command=self._reset_ui_colors, takefocus=0).pack(pady=10)

    # ──────────────────────────────────────────────────────────────────────────
    #  TAB: STATS
    # ──────────────────────────────────────────────────────────────────────────

    def _build_stats_tab(self):
        # ── Главный счётчик ───────────────────────────────────────────────────
        top_frame = tk.Frame(self.stats_tab)
        top_frame.pack(pady=(24, 4))

        # Пойманных — большой
        self.counter_var = tk.StringVar(value="0")
        tk.Label(top_frame, textvariable=self.counter_var,
                 font=("Helvetica", 56, "bold"), fg="#ffd700").pack()
        tk.Label(top_frame, text="🐟  поймано рыбы",
                 font=("Helvetica", 12), fg="#888").pack()

        # ── Статистика ────────────────────────────────────────────────────────
        stat_frame = tk.Frame(self.stats_tab)
        stat_frame.pack(pady=12, fill=tk.X, padx=30)

        # Сетка: caught / missed / accuracy
        for col, (label, varname, color) in enumerate([
            ("✅ Поймано",  "_stat_caught_var",   "#a6e3a1"),
            ("❌ Промахи",  "_stat_missed_var",   "#f38ba8"),
            ("🎯 Точность", "_stat_acc_var",      "#89b4fa"),
        ]):
            setattr(self, varname, tk.StringVar(value="0"))
            cell = tk.Frame(stat_frame, relief=tk.GROOVE, bd=1)
            cell.grid(row=0, column=col, padx=4, sticky="nsew")
            stat_frame.columnconfigure(col, weight=1)
            tk.Label(cell, text=label, font=("Helvetica", 8), fg="#888").pack(pady=(4,0))
            tk.Label(cell, textvariable=getattr(self, varname),
                     font=("Helvetica", 18, "bold"), fg=color).pack(pady=(0,4))

        # ── Последний результат ───────────────────────────────────────────────
        res_frame = tk.Frame(self.stats_tab, relief=tk.GROOVE, bd=1)
        res_frame.pack(fill=tk.X, padx=30, pady=6)
        tk.Label(res_frame, text="Последняя рыбалка:",
                 font=("Helvetica", 9), fg="#888").pack(side=tk.LEFT, padx=8, pady=6)
        self._last_result_label = tk.Label(res_frame, textvariable=self._last_result_var,
                                            font=("Helvetica", 10, "bold"))
        self._last_result_label.pack(side=tk.LEFT, padx=4)

        # ── История (лог последних 8 результатов) ────────────────────────────
        log_frame = tk.LabelFrame(self.stats_tab, text=" История ",
                                   font=("Helvetica", 8, "bold"), padx=6, pady=4)
        log_frame.pack(fill=tk.X, padx=30, pady=4)
        self._history_var = tk.StringVar(value="(пусто)")
        tk.Label(log_frame, textvariable=self._history_var,
                 font=("Courier", 9), justify=tk.LEFT, anchor="w").pack(fill=tk.X)

        # ── Кнопка сброса ─────────────────────────────────────────────────────
        self.reset_btn = tk.Button(self.stats_tab, text="↺  Сбросить всю статистику",
                                    command=self.reset_counter,
                                    font=("Helvetica", 10), padx=20, pady=5, takefocus=0)
        self.reset_btn.pack(pady=10)

        self._result_history: list[str] = []  # внутренний лог

    # ──────────────────────────────────────────────────────────────────────────
    #  UI COLORS
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_ui_colors(self):
        bg     = self.var_ui_bg.get()
        fg     = self.var_ui_fg.get()
        btn_bg = self.var_ui_button_bg.get()
        btn_fg = self.var_ui_button_fg.get()

        self.root.configure(bg=bg)
        for tab in [self.main_tab, self.settings_tab, self.ai_tab,
                    self.ui_tab, self.stats_tab]:
            tab.configure(bg=bg)

        self.status_label.configure(bg=bg, fg=self.var_ui_accent.get())
        self.ai_badge.configure(bg=bg)
        self.start_btn.configure(bg=btn_bg, fg=btn_fg)
        self.stop_btn.configure(bg=btn_bg, fg=btn_fg)
        self.reset_btn.configure(bg=btn_bg, fg=btn_fg)

        self.cfg.update({
            "ui_bg": bg, "ui_fg": fg,
            "ui_button_bg": btn_bg, "ui_button_fg": btn_fg,
            "ui_accent": self.var_ui_accent.get(),
            "ui_status_bg": self.var_ui_status_bg.get(),
        })
        save_config(self.cfg)

    def _reset_ui_colors(self):
        for k in ("ui_bg", "ui_fg", "ui_button_bg", "ui_button_fg",
                  "ui_accent", "ui_status_bg"):
            getattr(self, f"var_{k}").set(DEFAULTS[k])
        self._apply_ui_colors()
        for widget in self.ui_tab.winfo_children():
            widget.destroy()
        self._build_ui_tab()

    def _update_ai_badge(self):
        if self.var_ai_mode.get():
            self.ai_badge.config(text="🤖 AI  ON  —  Kalman + PID активен", fg="#a6e3a1")
        else:
            self.ai_badge.config(text="🤖 AI  OFF  —  классический режим", fg="#888")

    # ──────────────────────────────────────────────────────────────────────────
    #  CONFIG COLLECT
    # ──────────────────────────────────────────────────────────────────────────

    def _collect_cfg(self) -> dict:
        return {
            "hotkey":           self.var_hotkey.get(),
            "top":              self.var_top.get(),
            "left":             self.var_left.get(),
            "width":            self.var_width.get(),
            "height":           self.var_height.get(),
            "deadzone":         self.var_deadzone.get(),
            "auto_click_delay": self.var_ac_delay.get(),
            "yellow_h_min": 20, "yellow_h_max": 40,
            "yellow_s_min": 150, "yellow_v_min": 150,
            "safe_h_min": 80,   "safe_h_max": 100,
            "safe_s_min": 150,  "safe_v_min": 150,
            "ui_bg":            self.var_ui_bg.get(),
            "ui_fg":            self.var_ui_fg.get(),
            "ui_button_bg":     self.var_ui_button_bg.get(),
            "ui_button_fg":     self.var_ui_button_fg.get(),
            "ui_accent":        self.var_ui_accent.get(),
            "ui_status_bg":     self.var_ui_status_bg.get(),
            # AI
            "ai_mode":            self.var_ai_mode.get(),
            "pid_kp":             self.var_pid_kp.get(),
            "pid_ki":             self.var_pid_ki.get(),
            "pid_kd":             self.var_pid_kd.get(),
            "kalman_proc_noise":  self.var_kalman_proc.get(),
            "kalman_meas_noise":  self.var_kalman_meas.get(),
            "predict_frames":      self.var_predict_frames.get(),
            "min_success_ratio":   self.var_min_success_ratio.get(),
            "min_minigame_secs":   self.var_min_minigame_secs.get(),
        }

    def _save_all(self):
        self.cfg = self._collect_cfg()
        save_config(self.cfg)
        messagebox.showinfo("Сохранено", "Настройки сохранены в файл.")

    def _reset_defaults(self):
        if not messagebox.askyesno("Сброс", "Сбросить все настройки к умолчаниям?"):
            return
        self.cfg = dict(DEFAULTS)
        save_config(self.cfg)
        self._init_vars()
        self._register_hotkey(self.cfg["hotkey"])
        self._apply_ui_colors()
        self._update_ai_badge()
        for widget in self.ui_tab.winfo_children():
            widget.destroy()
        self._build_ui_tab()

    # ──────────────────────────────────────────────────────────────────────────
    #  PREVIEW LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def _preview_loop(self):
        if self.notebook.index(self.notebook.select()) == 1:
            region = {
                "top":    self.var_top.get(),
                "left":   self.var_left.get(),
                "width":  self.var_width.get(),
                "height": self.var_height.get(),
            }
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            pad = 120
            gt  = max(0, region["top"]  - pad)
            gl  = max(0, region["left"] - pad)
            gb  = min(sh, region["top"]  + region["height"] + pad)
            gr  = min(sw, region["left"] + region["width"]  + pad)
            grab = {"top": gt, "left": gl, "width": gr - gl, "height": gb - gt}
            iy1, ix1 = region["top"]  - gt, region["left"] - gl
            iy2, ix2 = iy1 + region["height"], ix1 + region["width"]
            try:
                img_bgra  = np.array(self.sct.grab(grab))
                img_rgb   = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGB)
                composite = (img_rgb * 0.35).astype(np.uint8)
                composite[iy1:iy2, ix1:ix2] = img_rgb[iy1:iy2, ix1:ix2]
                cv2.rectangle(composite, (ix1, iy1), (ix2, iy2), (0, 220, 80), 2)
                tw = 360
                th = max(10, int(360 * grab["height"] / grab["width"]))
                resized = cv2.resize(composite, (tw, th), interpolation=cv2.INTER_NEAREST)
                self.tk_image = ImageTk.PhotoImage(Image.fromarray(resized))
                self.preview_canvas.config(image=self.tk_image, text="")
            except Exception:
                self.preview_canvas.config(image='', text="Недопустимая область")
        self.root.after(150, self._preview_loop)

    # ──────────────────────────────────────────────────────────────────────────
    #  UI LOCK / UNLOCK
    # ──────────────────────────────────────────────────────────────────────────

    def _lock_ui(self):
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.reset_btn.config(state=tk.DISABLED)
        self.hotkey_btn.config(state=tk.DISABLED)
        for i in range(1, 5):
            self.notebook.tab(i, state="disabled")

    def _unlock_ui(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.reset_btn.config(state=tk.NORMAL)
        self.hotkey_btn.config(state=tk.NORMAL)
        for i in range(1, 5):
            self.notebook.tab(i, state="normal")

    # ──────────────────────────────────────────────────────────────────────────
    #  HOTKEY
    # ──────────────────────────────────────────────────────────────────────────

    def _register_hotkey(self, new_key: str, old_key: str = None):
        if old_key:
            try: keyboard.remove_hotkey(old_key.lower())
            except: pass
        try:
            keyboard.add_hotkey(new_key.lower(), self._toggle_from_hotkey)
        except Exception as e:
            messagebox.showerror("Ошибка хоткея",
                                  f"Не удалось зарегистрировать '{new_key}':\n{e}")

    def _begin_hotkey_capture(self):
        if self._hotkey_wait: return
        self._hotkey_wait = True
        self.hotkey_btn.config(state=tk.DISABLED, text="Нажми клавишу…")
        self.root.bind("<Key>", self._on_hotkey_key_pressed)

    def _on_hotkey_key_pressed(self, event):
        key = event.keysym.upper()
        if key in ['SHIFT', 'CTRL', 'ALT', 'CONTROL', 'MENU',
                   'CAPS_LOCK', 'NUM_LOCK', 'SCROLL_LOCK', 'ESCAPE']:
            return
        self.root.unbind("<Key>")
        self._hotkey_wait = False
        old_key = self.var_hotkey.get()
        self.var_hotkey.set(key)
        self.cfg["hotkey"] = key
        self._register_hotkey(key, old_key)
        save_config(self._collect_cfg())
        self.hotkey_btn.config(state=tk.NORMAL, text="Сменить")

    def _toggle_from_hotkey(self):
        self.root.after(0, self.toggle_bot)

    def toggle_bot(self):
        if self.bot_running: self.stop_bot()
        else:                self.start_bot()

    # ──────────────────────────────────────────────────────────────────────────
    #  BOT CONTROL
    # ──────────────────────────────────────────────────────────────────────────

    def start_bot(self):
        if self.bot_running: return
        self.cfg = self._collect_cfg()
        self.bot_running = True
        self._set_ui_state(True)
        self._lock_ui()
        self._update_ai_badge()
        self.root.focus_force()
        self.root.attributes('-topmost', True)
        self.bot_thread = threading.Thread(target=self._bot_loop, daemon=True)
        self.bot_thread.start()
        print("[DEBUG] Бот запущен")

    def stop_bot(self):
        self.bot_running = False
        self._set_ui_state(False)
        self._unlock_ui()
        release_keys()
        self._clear_telemetry()
        self.root.focus_force()
        print("[DEBUG] Бот остановлен")

    def _set_ui_state(self, running: bool):
        self.status_label.config(
            text="● RUNNING" if running else "● IDLE",
            fg="#27ae60" if running else "#888888"
        )

    def _record_result(self, caught: bool, in_zone_ratio: float, duration: float):
        """Вызывается из главного потока после каждой мини-игры."""
        if caught:
            self.fish_caught += 1
            label = f"✅ ПОЙМАНО  (в зоне {in_zone_ratio*100:.0f}%,  {duration:.1f}с)"
            self._last_result_var.set(label)
            self._last_result_label.config(fg="#a6e3a1")
        else:
            self.fish_missed += 1
            label = f"❌ ПРОМАХ   (в зоне {in_zone_ratio*100:.0f}%,  {duration:.1f}с)"
            self._last_result_var.set(label)
            self._last_result_label.config(fg="#f38ba8")

        # Обновить большой счётчик и статистику
        self.counter_var.set(str(self.fish_caught))
        self._stat_caught_var.set(str(self.fish_caught))
        self._stat_missed_var.set(str(self.fish_missed))
        total = self.fish_caught + self.fish_missed
        acc = (self.fish_caught / total * 100) if total > 0 else 0.0
        self._stat_acc_var.set(f"{acc:.0f}%")

        # История (последние 8)
        icon = "✅" if caught else "❌"
        self._result_history.append(
            f"{icon}  {in_zone_ratio*100:4.0f}%  {duration:4.1f}с"
        )
        if len(self._result_history) > 8:
            self._result_history.pop(0)
        self._history_var.set("\n".join(reversed(self._result_history)))

        print(f"[COUNTER] {'CAUGHT' if caught else 'MISSED'} | "
              f"ratio={in_zone_ratio:.2f} dur={duration:.1f}s | "
              f"total caught={self.fish_caught} missed={self.fish_missed}")

    # Legacy compat (используется в _lock_ui)
    def increment_counter(self):
        pass  # заменён на _record_result

    def reset_counter(self):
        self.fish_caught = 0
        self.fish_missed = 0
        self.counter_var.set("0")
        self._stat_caught_var.set("0")
        self._stat_missed_var.set("0")
        self._stat_acc_var.set("0%")
        self._last_result_var.set("—")
        self._last_result_label.config(fg="#cdd6f4")
        self._result_history.clear()
        self._history_var.set("(пусто)")
        print("[DEBUG] Статистика сброшена")

    def _smart_sleep(self, duration: float) -> bool:
        end = time.time() + duration
        hotkey = self.cfg["hotkey"].lower()
        while time.time() < end:
            if not self.bot_running or keyboard.is_pressed(hotkey):
                self.bot_running = False
                return True
            time.sleep(0.05)
        return False

    def _move_cursor_center(self):
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        pydirectinput.moveTo(sw // 2, sh // 2)

    def _clear_telemetry(self):
        for var in (self._tele_cursor_pos, self._tele_cursor_vel,
                    self._tele_safe_pos, self._tele_error,
                    self._tele_pid_out, self._tele_confidence):
            var.set("—")

    def _push_telemetry(self, c_filt, c_vel, s_filt, error, pid_out, conf):
        """Обновить телеметрию в главном потоке (вызов через root.after)."""
        self._tele_cursor_pos.set(f"{c_filt:+.1f} px")
        self._tele_cursor_vel.set(f"{c_vel:+.2f} px/fr")
        self._tele_safe_pos.set(f"{s_filt:+.1f} px")
        self._tele_error.set(f"{error:+.1f} px")
        self._tele_pid_out.set(f"{pid_out:+.3f}")
        self._tele_confidence.set(f"{conf:.0f}%")

    # ──────────────────────────────────────────────────────────────────────────
    #  BOT LOOP  ★  главный цикл
    # ──────────────────────────────────────────────────────────────────────────

    def _bot_loop(self):
        cfg      = self.cfg
        hotkey   = cfg["hotkey"].lower()
        deadzone = cfg["deadzone"]
        ac_delay = cfg["auto_click_delay"]
        ai_mode  = cfg.get("ai_mode", True)
        region   = {"top": cfg["top"], "left": cfg["left"],
                    "width": cfg["width"], "height": cfg["height"]}

        # ── Инициализация AI ──────────────────────────────────────────────────
        kalman_cursor = KalmanFilter1D(
            process_noise=cfg.get("kalman_proc_noise", 2.0),
            measurement_noise=cfg.get("kalman_meas_noise", 8.0),
        )
        kalman_safe = KalmanFilter1D(
            process_noise=cfg.get("kalman_proc_noise", 2.0),
            measurement_noise=cfg.get("kalman_meas_noise", 8.0),
        )
        pid = PIDController(
            kp=cfg.get("pid_kp", 0.8),
            ki=cfg.get("pid_ki", 0.05),
            kd=cfg.get("pid_kd", 0.30),
            output_limits=(-1.0, 1.0),
        )
        predict_frames = int(cfg.get("predict_frames", 2))

        # Скользящее окно для Confidence (% кадров с успешной детекцией)
        CONF_WINDOW = 30
        det_history: list[int] = []

        state           = "BEFORE_FISHING"
        last_f_press    = 0.0
        last_auto_click = 0.0
        frames_lost     = 0
        prev_time       = time.time()

        # ── Умный счётчик ─────────────────────────────────────────────────────
        min_success_ratio = cfg.get("min_success_ratio", 0.30)
        min_minigame_secs = cfg.get("min_minigame_secs", 2.0)
        minigame_start    = 0.0
        in_zone_frames    = 0
        total_mg_frames   = 0

        sct = mss.MSS()
        try:
            while self.bot_running:
                if keyboard.is_pressed(hotkey):
                    self.root.after(0, self.stop_bot)
                    break

                now = time.time()
                dt  = max(now - prev_time, 1e-4)
                prev_time = now

                c_x, s_x = get_positions(sct, region, cfg)

                # Confidence
                det_history.append(1 if (c_x is not None and s_x is not None) else 0)
                if len(det_history) > CONF_WINDOW:
                    det_history.pop(0)
                confidence = sum(det_history) / len(det_history) * 100.0

                # Auto-click
                if now - last_auto_click >= ac_delay:
                    pydirectinput.click()
                    last_auto_click = now

                # ══ ПЕРЕД РЫБАЛКОЙ ═══════════════════════════════════════════
                if state == "BEFORE_FISHING":
                    self._move_cursor_center()
                    kalman_cursor.reset()
                    kalman_safe.reset()
                    pid.reset()

                    if now - last_f_press > 1.5:
                        pydirectinput.press('f')
                        last_f_press = now

                    if s_x is not None and c_x is not None:
                        state = "MINIGAME"
                        frames_lost    = 0
                        minigame_start = now
                        in_zone_frames = 0
                        total_mg_frames = 0
                        print("[DEBUG] → MINIGAME")
                    else:
                        time.sleep(0.05)

                elif state == "MINIGAME":
                    if c_x is not None and s_x is not None:
                        frames_lost = 0
                        total_mg_frames += 1

                        if ai_mode:
                            c_filt = kalman_cursor.update(c_x)
                            s_filt = kalman_safe.update(s_x)
                            c_pred = kalman_cursor.predict_ahead(predict_frames)
                            c_vel  = kalman_cursor.velocity
                            error  = c_pred - s_filt
                            pid_out = pid.compute(error, dt)

                            # Трекинг: курсор внутри зоны?
                            if abs(error) <= deadzone:
                                in_zone_frames += 1

                            self.root.after(0, self._push_telemetry,
                                            c_filt, c_vel, s_filt,
                                            error, pid_out, confidence)

                            if abs(error) <= deadzone:
                                release_keys()
                            elif error > 0:
                                pydirectinput.keyUp('d')
                                pydirectinput.keyDown('a')
                                if abs(error) < 40:
                                    time.sleep(max(0.004, 0.025 * abs(pid_out)))
                                    pydirectinput.keyUp('a')
                            else:
                                pydirectinput.keyUp('a')
                                pydirectinput.keyDown('d')
                                if abs(error) < 40:
                                    time.sleep(max(0.004, 0.025 * abs(pid_out)))
                                    pydirectinput.keyUp('d')

                        else:
                            dist = abs(c_x - s_x)
                            # Трекинг и для классического режима
                            if dist <= deadzone:
                                in_zone_frames += 1

                            if c_x < (s_x - deadzone):
                                pydirectinput.keyUp('a')
                                pydirectinput.keyDown('d')
                                if dist < 40:
                                    time.sleep(0.01)
                                    pydirectinput.keyUp('d')
                            elif c_x > (s_x + deadzone):
                                pydirectinput.keyUp('d')
                                pydirectinput.keyDown('a')
                                if dist < 40:
                                    time.sleep(0.01)
                                    pydirectinput.keyUp('a')
                            else:
                                release_keys()

                    else:
                        frames_lost += 1
                        if frames_lost > 10:
                            release_keys()
                            # ── Подсчёт качества мини-игры ──────────────────
                            mg_duration  = now - minigame_start
                            in_zone_ratio = (in_zone_frames / total_mg_frames
                                             if total_mg_frames > 0 else 0.0)
                            caught = (mg_duration  >= min_minigame_secs and
                                      in_zone_ratio >= min_success_ratio)
                            # Передаём результат в UI
                            self.root.after(0, self._record_result,
                                            caught, in_zone_ratio, mg_duration)
                            state = "REWARD"
                            print(f"[DEBUG] → REWARD | dur={mg_duration:.1f}s "
                                  f"ratio={in_zone_ratio:.2f} caught={caught}")

                elif state == "REWARD":
                    if self._smart_sleep(2.0):
                        break
                    self._move_cursor_center()
                    time.sleep(3)
                    pydirectinput.click()
                    # Результат уже записан при переходе в REWARD
                    if self._smart_sleep(2.0):
                        break
                    state = "BEFORE_FISHING"
                    last_f_press = now
                    print("[DEBUG] → BEFORE_FISHING")

        finally:
            release_keys()
            self.root.after(0, lambda: self._set_ui_state(False))
            self.root.after(0, self._clear_telemetry)

    # ──────────────────────────────────────────────────────────────────────────
    #  CLOSE
    # ──────────────────────────────────────────────────────────────────────────

    def _on_closing(self):
        self.bot_running = False
        release_keys()
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = FishingBotGUI(root)
    root.mainloop()
