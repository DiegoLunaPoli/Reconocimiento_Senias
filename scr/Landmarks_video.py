#!/usr/bin/env python3
"""
record_landmarks.py
Graba hasta 2 minutos desde la webcam, extrae los 21 landmarks por mano usando MediaPipe
y guarda un CSV por mano por fotograma en la carpeta 'dataset_landmarks'.

Controles en la ventana:
 - Presiona 's' para empezar a grabar.
 - Presiona 'q' para detener antes de los 2 minutos.
"""

import cv2
import mediapipe as mp
import os
import csv
import time
from datetime import datetime

# ---------- Config ----------
OUTPUT_DIR = "dataset_landmarks"
MAX_SECONDS = 120  # 2 minutos
VIDEO_SAVE = True   # guarda también un video .mp4 de la sesión
VIDEO_CODEC = "mp4v"  # codec para VideoWriter
# ----------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_landmarks_csv(path, frame_idx, ts, hand_label, landmarks):
    """
    landmarks: lista de 21 tuples (x,y,z) (normalizados)
    Guarda un CSV con columnas:
    frame,timestamp,hand,landmark_0_x,landmark_0_y,landmark_0_z, ..., landmark_20_z
    """
    header = ["frame", "timestamp", "hand"]
    for i in range(21):
        header += [f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"]

    row = [frame_idx, f"{ts:.6f}", hand_label]
    for (x, y, z) in landmarks:
        row += [f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"]

    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)

def main():
    # Preguntar la letra
    letra = input("Escribe la letra (dactilológico, ej. 'a', 'b', 'ñ', ...) que vas a representar: ").strip()
    if letra == "":
        print("No especificaste letra. Saliendo.")
        return
    letra = letra.replace(" ", "_").lower()

    ensure_dir(OUTPUT_DIR)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    # Intentar leer FPS de la cámara; si no está disponible, usar 30.
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap_fps or cap_fps <= 0:
        cap_fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    video_writer = None
    start_time = None
    recording = False
    frame_idx = 0
    saved_files = 0
    start_timestamp_str = timestamp_str()

    if VIDEO_SAVE:
        out_video_path = os.path.join(OUTPUT_DIR, f"{letra}_{start_timestamp_str}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        video_writer = cv2.VideoWriter(out_video_path, fourcc, cap_fps, (frame_width, frame_height))

    # MediaPipe hands init
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("Cámara abierta. Presiona 's' para empezar la grabación (máx 120s). Presiona 'q' para salir.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error leyendo cámara.")
                break

            # Flip para modo espejo (opcional)
            frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image_for_draw = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Si estamos grabando, procesar y guardar landmarks
            if recording:
                elapsed = time.time() - start_time
                if elapsed >= MAX_SECONDS:
                    print("Tiempo máximo alcanzado.")
                    recording = False

                # procesar frame
                results = hands.process(frame_rgb)

                # dibujar landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        mp_drawing.draw_landmarks(
                            image_for_draw,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0,128,255), thickness=2)
                        )

                        # extraer coordenadas normalizadas x,y,z (21 puntos)
                        coords = []
                        for lm in hand_landmarks.landmark:
                            coords.append((lm.x, lm.y, lm.z))

                        # handedness classification label (e.g., 'Left' or 'Right')
                        hand_label = handedness.classification[0].label if handedness.classification else "Unknown"

                        # guardado CSV
                        ts = time.time()
                        fname = f"{letra}_{start_timestamp_str}_frame{frame_idx:06d}_{hand_label.upper()}.csv"
                        fpath = os.path.join(OUTPUT_DIR, fname)
                        save_landmarks_csv(fpath, frame_idx, ts, hand_label.upper(), coords)
                        saved_files += 1

                # escribir video
                if video_writer is not None:
                    # recordar que flip invertimos antes, así mantenemos espejo en el video también
                    video_writer.write(image_for_draw)

                # mostrar tiempo restante en la imagen
                remaining = max(0, MAX_SECONDS - int(elapsed))
                cv2.putText(image_for_draw, f"Grabando ({remaining}s restantes) - Frame: {frame_idx}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

                frame_idx += 1
            else:
                cv2.putText(image_for_draw, "Presiona 's' para iniciar grabacion, 'q' para salir",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Record Landmarks - Presiona s para empezar", image_for_draw)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not recording:
                # iniciar grabación
                recording = True
                start_time = time.time()
                frame_idx = 0
                start_timestamp_str = timestamp_str()
                print("INICIANDO grabación... (presiona 'q' para terminar antes de 120s)")
            elif key == ord('q'):
                print("Detenido por el usuario.")
                break

    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        hands.close()
        cv2.destroyAllWindows()
        print(f"Sesión finalizada. Archivos CSV guardados en '{OUTPUT_DIR}': {saved_files}")

if __name__ == "__main__":
    main()
