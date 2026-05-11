import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import cv2
import numpy as np
import mss
import ctypes
import sys
import os
import json
import keyboard

# ПРОВЕРКА ПРАВ АДМИНИСТРАТОРА
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    print("❌ Нужны права администратора!")
    try:
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
        sys.exit()
    except:
        messagebox.showerror("Ошибка", "Запусти от имени администратора!")
        sys.exit()

# Глобальные переменные
running = False
auto_detected = False
scanner_active = True

# ФАЙЛ НАСТРОЕК
CONFIG_FILE = "fishing_bot_config.json"

# ТОЧНАЯ НАСТРОЙКА ЦВЕТА #2ac7ac
GREEN_LOWER = np.array([80, 120, 120])
GREEN_UPPER = np.array([90, 255, 230])

# Настройки по умолчанию
DEAD_ZONE = 20
MOVE_COOLDOWN = 0.05
MIN_AREA = 60
MAX_AREA = 5000

# Область захвата
monitor = {
    "left": 0,
    "top": 0,
    "width": 1920,
    "height": 1080
}

# Цвета GUI (используем цвета которые НЕ похожи на #2ac7ac)
BG = "#1a1a2e"      # тёмно-синий
CARD = "#16213e"    # тёмно-синий
TEXT = "#e0e0e0"    # серый
PINK = "#ff6b6b"    # розовый
BUTTON = "#0f3460"  # тёмно-синий

def log(msg):
    log_box.insert(tk.END, msg + "\n")
    log_box.see(tk.END)

# НАДЁЖНАЯ ЭМУЛЯЦИЯ КЛАВИШ
def press_key(key):
    KEY_CODES = {'a': 0x41, 'd': 0x44}
    key_code = KEY_CODES.get(key)
    if not key_code:
        return
    ctypes.windll.user32.keybd_event(key_code, 0, 0, 0)
    time.sleep(0.03)
    ctypes.windll.user32.keybd_event(key_code, 0, 2, 0)

def press_a():
    press_key('a')
    log("🔴 ЛЕВО")
    direction_label.config(text="◀ ЛЕВО")

def press_d():
    press_key('d')
    log("🟢 ПРАВО")
    direction_label.config(text="ПРАВО ▶")

# ========== БЛОКИРОВКА/РАЗБЛОКИРОВКА ИНТЕРФЕЙСА ==========
def lock_ui():
    left_entry.config(state="disabled")
    top_entry.config(state="disabled")
    width_entry.config(state="disabled")
    height_entry.config(state="disabled")
    apply_coords_btn.config(state="disabled")
    sensitivity_btn.config(state="disabled")
    dead_zone_slider.config(state="disabled")
    test_btn.config(state="disabled")
    area_frame.config(fg="gray")
    sens_frame.config(fg="gray")
    left_entry.config(bg="#2a2a3e", fg="gray")
    top_entry.config(bg="#2a2a3e", fg="gray")
    width_entry.config(bg="#2a2a3e", fg="gray")
    height_entry.config(bg="#2a2a3e", fg="gray")

def unlock_ui():
    left_entry.config(state="normal")
    top_entry.config(state="normal")
    width_entry.config(state="normal")
    height_entry.config(state="normal")
    apply_coords_btn.config(state="normal")
    sensitivity_btn.config(state="normal")
    dead_zone_slider.config(state="normal")
    test_btn.config(state="normal")
    area_frame.config(fg=PINK)
    sens_frame.config(fg=PINK)
    left_entry.config(bg=CARD, fg=TEXT)
    top_entry.config(bg=CARD, fg=TEXT)
    width_entry.config(bg=CARD, fg=TEXT)
    height_entry.config(bg=CARD, fg=TEXT)

# ========== ТЕСТ ЦВЕТА ==========
def test_color_detection():
    """Тест: проверяет, видит ли камера цвет #2ac7ac на экране"""
    log("🔬 Тест обнаружения цвета #2ac7ac...")
    
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        img = np.array(screenshot)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
        
        green_pixels = np.sum(mask > 0)
        log(f"📊 Найдено пикселей цвета #2ac7ac: {green_pixels}")
        
        if green_pixels > 100:
            log("✅ Цвет #2ac7ac обнаружен! Полоска есть на экране")
            return True
        else:
            log("❌ Цвет #2ac7ac НЕ обнаружен!")
            log("👉 Убедись что в игре видна полоска")
            return False

# ========== ФОНОВЫЙ СКАНЕР (С ПРИОРИТЕТОМ) ==========
scanner_thread = None
scanner_run = True

def start_scanner():
    global scanner_thread, scanner_run
    scanner_run = True
    scanner_thread = threading.Thread(target=scanner_loop, daemon=True)
    scanner_thread.start()
    log("🔄 Сканер запущен")

def stop_scanner():
    global scanner_run
    scanner_run = False

def scanner_loop():
    global auto_detected, monitor
    
    with mss.mss() as sct:
        scan_count = 0
        last_found_time = 0
        
        while scanner_run:
            try:
                scan_count += 1
                
                # Захват экрана
                screenshot = sct.grab(sct.monitors[1])
                img = np.array(screenshot)
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Поиск цвета
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
                
                # Фильтрация шума
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                found = False
                
                if contours:
                    valid_contours = []
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if MIN_AREA < area < MAX_AREA:
                            valid_contours.append(cnt)
                    
                    if valid_contours:
                        largest = max(valid_contours, key=cv2.contourArea)
                        area = cv2.contourArea(largest)
                        x, y, w, h = cv2.boundingRect(largest)
                        
                        # Проверка пропорций (полоска должна быть горизонтальной)
                        if w > h * 1.5:
                            found = True
                            padding = 40
                            new_left = max(0, x - padding)
                            new_top = max(0, y - padding)
                            new_width = min(sct.monitors[1]["width"] - new_left, w + padding * 2)
                            new_height = min(sct.monitors[1]["height"] - new_top, h + padding * 2)
                            
                            if not auto_detected or abs(monitor["left"] - new_left) > 50:
                                monitor = {
                                    "left": new_left,
                                    "top": new_top,
                                    "width": new_width,
                                    "height": new_height
                                }
                                auto_detected = True
                                last_found_time = time.time()
                                
                                # Обновляем GUI
                                root.after(0, lambda: left_var.set(str(monitor["left"])))
                                root.after(0, lambda: top_var.set(str(monitor["top"])))
                                root.after(0, lambda: width_var.set(str(monitor["width"])))
                                root.after(0, lambda: height_var.set(str(monitor["height"])))
                                
                                log(f"✅ [СКАН #{scan_count}] Полоска найдена! Размер: {w}x{h}")
                
                # Каждые 30 сканов пишем статус если не найдено
                if scan_count % 30 == 0 and not auto_detected:
                    log(f"🔍 Скан #{scan_count}: ожидание полоски...")
                
                time.sleep(0.5)
                
            except Exception as e:
                log(f"❌ Ошибка сканера: {e}")
                time.sleep(1)

# ========== ОСНОВНОЙ БОТ ==========
bot_thread = None
bot_run = False

def start_bot():
    global bot_run, running
    if bot_run:
        return
    
    if not auto_detected:
        log("⚠ Полоска ещё не найдена!")
        return
    
    bot_run = True
    running = True
    bot_thread = threading.Thread(target=bot_loop, daemon=True)
    bot_thread.start()
    
    lock_ui()
    
    log("✅ БОТ ЗАПУЩЕН! (F8 - СТОП)")
    status_label.config(text="БОТ РАБОТАЕТ", fg="green")
    start_btn.config(text="СТОП (F8)", bg="#dc143c")
    direction_label.config(text="БОТ АКТИВЕН", fg="green")

def stop_bot():
    global bot_run, running
    bot_run = False
    running = False
    
    unlock_ui()
    
    log("⏹ БОТ ОСТАНОВЛЕН (ESC)")
    status_label.config(text="БОТ ОСТАНОВЛЕН", fg=PINK)
    start_btn.config(text="СТАРТ (F8)", bg=PINK)
    direction_label.config(text="ОСТАНОВЛЕН", fg=PINK)
    info_label.config(text="📍 Нажми F8 для старта")

def toggle_bot():
    if bot_run:
        stop_bot()
    else:
        start_bot()

def bot_loop():
    with mss.mss() as sct:
        last_move = 0
        
        while bot_run:
            try:
                img = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                if frame.size == 0:
                    time.sleep(0.05)
                    continue
                
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
                
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                center_x = monitor["width"] // 2
                now = time.time()
                
                if contours:
                    valid_contours = [cnt for cnt in contours if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA]
                    
                    if valid_contours:
                        largest = max(valid_contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest)
                        green_center = x + w // 2
                        
                        root.after(0, lambda: info_label.config(text=f"📍 Позиция: {green_center}px"))
                        
                        if now - last_move >= MOVE_COOLDOWN:
                            if green_center < center_x - DEAD_ZONE:
                                press_a()
                                last_move = now
                            elif green_center > center_x + DEAD_ZONE:
                                press_d()
                                last_move = now
                            else:
                                root.after(0, lambda: direction_label.config(text="● ЦЕНТР"))
                    else:
                        root.after(0, lambda: info_label.config(text="⚠ Нет полоски"))
                else:
                    root.after(0, lambda: info_label.config(text="❌ Полоска не найдена"))
                    root.after(0, lambda: direction_label.config(text="ПОИСК..."))
                    
            except Exception as e:
                pass
            
            time.sleep(0.02)

def save_config():
    config = {"monitor": monitor, "dead_zone": DEAD_ZONE}
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f)
    except:
        pass

def load_config():
    global monitor, DEAD_ZONE
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
            monitor = config.get("monitor", monitor)
            DEAD_ZONE = config.get("dead_zone", DEAD_ZONE)
            left_var.set(str(monitor["left"]))
            top_var.set(str(monitor["top"]))
            width_var.set(str(monitor["width"]))
            height_var.set(str(monitor["height"]))
            dead_zone_var.set(DEAD_ZONE)
    except:
        pass

def apply_manual_coords():
    global monitor, auto_detected
    if bot_run:
        log("❌ Нельзя менять во время работы бота!")
        return
    try:
        monitor = {
            "left": int(left_var.get()),
            "top": int(top_var.get()),
            "width": int(width_var.get()),
            "height": int(height_var.get())
        }
        auto_detected = True
        log(f"📌 Координаты установлены")
        save_config()
    except ValueError:
        log("❌ Ошибка!")

def apply_sensitivity():
    global DEAD_ZONE
    if bot_run:
        log("❌ Нельзя менять во время работы бота!")
        return
    DEAD_ZONE = dead_zone_var.get()
    log(f"⚙ Чувствительность: {DEAD_ZONE}px")
    save_config()

# ========== GUI ==========
root = tk.Tk()
root.title("Neverness to Everness - Autofishing Bot")
root.geometry("650x750")
root.configure(bg=BG)

# Заголовок
title = tk.Label(root, text="🐟 AUTOFISHING BOT", font=("Consolas", 24, "bold"), bg=BG, fg=PINK)
title.pack(pady=10)

status_label = tk.Label(root, text="СКАНЕР АКТИВЕН", bg=BG, fg=PINK, font=("Consolas", 10, "bold"))
status_label.pack()

# ========== ПРЕВЬЮ УБРАНО! ==========
# (Нет больше зелёной области которая детектится)

# ========== КНОПКИ ==========
test_btn = tk.Button(root, text="🔬 ТЕСТ ОБНАРУЖЕНИЯ ЦВЕТА #2ac7ac", 
                      command=test_color_detection,
                      bg=BUTTON, fg=TEXT, font=("Consolas", 10, "bold"))
test_btn.pack(pady=5)

# ========== ИНФОРМАЦИЯ ==========
info_frame = tk.LabelFrame(root, text="ИНФОРМАЦИЯ", bg=BG, fg=PINK, font=("Consolas", 12, "bold"))
info_frame.pack(fill="x", padx=20, pady=10)

tk.Label(info_frame, text="🎯 Цвет полоски: #2ac7ac (бирюзовый)", bg=BG, fg="#2ac7ac", font=("Consolas", 9, "bold")).pack(pady=2)
tk.Label(info_frame, text="⌨ F8 - СТАРТ / СТОП бота", bg=BG, fg=TEXT).pack(pady=2)
tk.Label(info_frame, text="🔬 'ТЕСТ' - проверить видит ли программа полоску", bg=BG, fg=TEXT).pack(pady=2)

# ========== КООРДИНАТЫ ==========
area_frame = tk.LabelFrame(root, text="ОБЛАСТЬ ПОЛОСКИ (определяется автоматически)", bg=BG, fg=PINK, font=("Consolas", 11, "bold"))
area_frame.pack(fill="x", padx=20, pady=10)

coord_grid = tk.Frame(area_frame, bg=BG)
coord_grid.pack(pady=10)

tk.Label(coord_grid, text="LEFT:", bg=BG, fg=TEXT).grid(row=0, column=0, padx=10)
left_var = tk.StringVar(value="0")
left_entry = tk.Entry(coord_grid, textvariable=left_var, width=8, bg=CARD, fg=TEXT, font=("Consolas", 10))
left_entry.grid(row=0, column=1, padx=10)

tk.Label(coord_grid, text="TOP:", bg=BG, fg=TEXT).grid(row=0, column=2, padx=10)
top_var = tk.StringVar(value="0")
top_entry = tk.Entry(coord_grid, textvariable=top_var, width=8, bg=CARD, fg=TEXT, font=("Consolas", 10))
top_entry.grid(row=0, column=3, padx=10)

tk.Label(coord_grid, text="WIDTH:", bg=BG, fg=TEXT).grid(row=1, column=0, padx=10)
width_var = tk.StringVar(value="0")
width_entry = tk.Entry(coord_grid, textvariable=width_var, width=8, bg=CARD, fg=TEXT, font=("Consolas", 10))
width_entry.grid(row=1, column=1, padx=10)

tk.Label(coord_grid, text="HEIGHT:", bg=BG, fg=TEXT).grid(row=1, column=2, padx=10)
height_var = tk.StringVar(value="0")
height_entry = tk.Entry(coord_grid, textvariable=height_var, width=8, bg=CARD, fg=TEXT, font=("Consolas", 10))
height_entry.grid(row=1, column=3, padx=10)

apply_coords_btn = tk.Button(area_frame, text="📌 ПРИМЕНИТЬ ВРУЧНУЮ", 
                              command=apply_manual_coords,
                              bg=BUTTON, fg=TEXT, font=("Consolas", 10, "bold"))
apply_coords_btn.pack(pady=5)

# ========== НАСТРОЙКИ ==========
sens_frame = tk.LabelFrame(root, text="ЧУВСТВИТЕЛЬНОСТЬ", bg=BG, fg=PINK, font=("Consolas", 11, "bold"))
sens_frame.pack(fill="x", padx=20, pady=10)

tk.Label(sens_frame, text="Мёртвая зона (5-100):", bg=BG, fg=TEXT).pack()
dead_zone_var = tk.IntVar(value=DEAD_ZONE)
dead_zone_slider = tk.Scale(sens_frame, from_=5, to=100, orient="horizontal", variable=dead_zone_var, bg=BG, length=400)
dead_zone_slider.pack()

sensitivity_btn = tk.Button(sens_frame, text="✅ ПРИМЕНИТЬ", 
                             command=apply_sensitivity,
                             bg=BUTTON, fg=TEXT, font=("Consolas", 10, "bold"))
sensitivity_btn.pack(pady=5)

# ========== УПРАВЛЕНИЕ ==========
direction_label = tk.Label(root, text="ОЖИДАНИЕ", bg=BG, fg=PINK, font=("Consolas", 20, "bold"))
direction_label.pack(pady=10)

info_label = tk.Label(root, text="📍 Нажми F8 для старта", bg=BG, fg=TEXT, font=("Consolas", 10))
info_label.pack()

btn_ctrl = tk.Frame(root, bg=BG)
btn_ctrl.pack(pady=10)

start_btn = tk.Button(btn_ctrl, text="СТАРТ (F8)", 
                      command=toggle_bot,
                      bg=PINK, fg="white", font=("Consolas", 14, "bold"), width=12)
start_btn.pack(side="left", padx=10)

# Цвет полоски (просто информационная метка, НЕ зелёная)
color_info = tk.Label(root, text="🎨 ЦВЕТ ПОЛОСКИ: #2ac7ac", bg=BG, fg="#2ac7ac", font=("Consolas", 9, "bold"))
color_info.pack(pady=5)

# ЛОГ
log_box = tk.Text(root, height=9, bg=CARD, fg=TEXT, font=("Consolas", 9))
log_box.pack(fill="both", padx=20, pady=10)

# ВОДЯНОЙ ЗНАК
watermark = tk.Label(root, text="https://t.me/losestrik1337", bg=BG, fg="#ff4444", font=("Consolas", 9, "bold"))
watermark.place(relx=1.0, rely=1.0, anchor="se", x=-10, y=-10)

# ========== ГОРЯЧИЕ КЛАВИШИ ==========
keyboard.add_hotkey("F8", toggle_bot)

# ========== ЗАПУСК ==========
load_config()
start_scanner()

log("🚀 Программа запущена!")
log("🎯 Цвет полоски: #2ac7ac")
log("🔬 Нажми 'ТЕСТ' чтобы проверить видит ли программа полоску")
log("🎮 Открой игру, бот сам найдёт полоску")
log("⌨ F8 - СТАРТ/СТОП")

root.mainloop()

stop_scanner()
stop_bot()