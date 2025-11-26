#!/usr/bin/env python3
"""
videos_to_landmarks.py

Procesa todos los videos organizados en carpetas por letra (videos_proc/<LETTER>/)
- Normaliza brillo, reduce ruido y redimensiona cada frame.
- Extrae landmarks (MediaPipe Hands) por frame.
- Agrega filas al CSV correspondiente en dataset_landmarks/<LETTER>.csv
  (si el CSV no existe lo crea con encabezado; si existe, agrega filas al final).

Formato de salida CSV (fila por detección):
 time,capture_type,x0,y0,z0,...,x20,y20,z20

Parámetros editables (al inicio del archivo):
 - VIDEOS_DIR: carpeta donde están las subcarpetas por letra con clips
 - DATASET_DIR: carpeta donde están/estarán los CSV por letra
 - FRAME_STEP: procesar cada N-ésimo frame (1 = todos)
 - TARGET_SIZE: (w,h) para redimensionar
 - SAVE_IMAGES: guardar frames procesados (opcional)
"""

import os
import csv
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# ---------------------- CONFIGURACIÓN ----------------------
VIDEOS_DIR   = "videos_proc"           # <-- carpeta con subcarpetas por letra
DATASET_DIR  = "dataset_landmarks"    # <-- carpeta con CSV por letra (existing)
FRAME_STEP   = 3                       # procesar cada N-ésimo frame (reduce carga)
TARGET_SIZE  = (640, 480)              # ancho, alto (resize)
DENoISE      = True                    # aplicar denoise
SAVE_IMAGES  = False                   # guardar frames procesados
IMAGES_DIR   = "debug_frames"          # si SAVE_IMAGES=True
CAPTURE_TYPE = "video"                 # valor en columna capture_type
# -----------------------------------------------------------

# MediaPipe config
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# header expected in CSVs
LM_HEADER = ["time", "capture_type"] + [f"{c}{i}" for i in range(21) for c in ("x","y","z")]

# ---------------------- UTILIDADES DE PREPROCESS ----------------------
def resize_frame(frame, target_size):
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

def denoise_frame(frame):
    # fastNlMeansDenoisingColored requiere imágenes BGR uint8
    return cv2.fastNlMeansDenoisingColored(frame, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

def equalize_brightness(frame_bgr):
    
    # Convertir a YCrCb, equalize Y (luminance), volver a BGR
    img_y_cr_cb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    bgr_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    return bgr_eq

def preprocess_frame(frame_bgr):
    # frame_bgr: BGR uint8
    f = frame_bgr
    # Denoise
    if DENoISE:
        try:
            f = denoise_frame(f)
        except Exception:
            pass
    # Brightness equalization
    try:
        f = equalize_brightness(f)
    except Exception:
        pass
    # Resize
    f = resize_frame(f, TARGET_SIZE)
    return f

# ---------------------- CSV APPEND ----------------------
def ensure_csv_with_header(csv_path):
    # Si no existe el CSV, crea y escribe header
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(LM_HEADER)

def append_landmark_row(csv_path, time_str, capture_type, landmark_list):
    """
    landmark_list: list/iterable de 63 floats (x0,y0,z0, ..., x20,y20,z20)
    """
    ensure_csv_with_header(csv_path)
    row = [time_str, capture_type] + [f"{v:.9f}" for v in landmark_list]
    with open(csv_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)

# ---------------------- MAIN PROCESS ----------------------
def process_video_file(video_path, out_csv_path, save_images=False, debug_folder=None, frame_step=1):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("ERROR: no se pudo abrir:", video_path)
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    saved = 0
    pbar = tqdm(total=total_frames//frame_step + 1, desc=Path(video_path).name, unit="step")
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        # Preprocesamiento
        proc = preprocess_frame(frame)

        # Convertir a RGB para MediaPipe
        image_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            # tomar la primera mano
            lm = results.multi_hand_landmarks[0]
            # construir lista 63 floats
            lm_list = []
            for l in lm.landmark:
                lm_list.extend([l.x, l.y, l.z])

            # tiempo: timestamp del video en segundos + datetime (ISO) para trazabilidad
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            time_str = f"{now_iso}|{pos_ms/1000:.3f}s"

            append_landmark_row(out_csv_path, time_str, CAPTURE_TYPE, lm_list)
            saved += 1
            saved_idx += 1

            if save_images and debug_folder:
                os.makedirs(debug_folder, exist_ok=True)
                out_img = os.path.join(debug_folder, f"{Path(video_path).stem}_f{frame_idx:06d}.jpg")
                cv2.imwrite(out_img, proc)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    hands.close()
    cap.release()
    return saved

def process_all(videos_dir=VIDEOS_DIR, dataset_dir=DATASET_DIR, frame_step=FRAME_STEP, save_images=SAVE_IMAGES):
    videos_dir = Path(videos_dir)
    dataset_dir = Path(dataset_dir)

    if save_images:
        Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    total_added = 0
    # iterar subdirs (cada subdir = label)
    for letter_folder in sorted(videos_dir.iterdir()):
        if not letter_folder.is_dir():
            continue
        letter = letter_folder.name
        csv_path = dataset_dir / f"{letter}.csv"
        ensure_csv_with_header(str(csv_path))

        # procesar cada video en esa carpeta (ordenado)
        video_files = sorted([p for p in letter_folder.glob("*") if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi")])
        if not video_files:
            print(f"[{letter}] no hay videos en {letter_folder}, saltando.")
            continue

        print(f"\n=== Procesando letra '{letter}' → {len(video_files)} archivos ===")
        # procesar cada archivo y añadir
        for vf in video_files:
            print(f"Procesando archivo: {vf.name}")
            debug_folder = None
            if save_images:
                debug_folder = os.path.join(IMAGES_DIR, letter)
            added = process_video_file(str(vf), str(csv_path), save_images, debug_folder, frame_step)
            print(f"  => Añadidas {added} filas al CSV {csv_path.name}")
            total_added += added

    print(f"\nProceso finalizado. Total filas añadidas: {total_added}")

# ---------------------- RUN ----------------------
if __name__ == "__main__":
    print("INFO: videos_dir:", VIDEOS_DIR)
    print("INFO: dataset_dir:", DATASET_DIR)
    print("INFO: frame_step:", FRAME_STEP, "target_size:", TARGET_SIZE, "denoise:", DENoISE)
    process_all()
