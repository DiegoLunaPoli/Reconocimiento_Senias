"""
capture_landmarks.py
Script para capturar landmarks de MediaPipe Hands en modo híbrido:
- Captura manual con SPACE
- Captura automática cada `capture_interval` segundos
- Guarda filas en dataset_landmarks/<gesture>.csv
- Muestra un pequeño contador en la esquina con tiempo faltante y total guardados
"""

import cv2
import mediapipe as mp
import time
import os
import csv

# -------------------------
# Config
# -------------------------
DATA_DIR = "dataset_landmarks"    # Carpeta donde se guardan los CSV por gesto
os.makedirs(DATA_DIR, exist_ok=True)

capture_interval = 5              # segundos entre capturas automáticas
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
max_num_hands = 1                 # para simplificar (puedes cambiarlo)

# Si quieres también guardar la imagen (pantallazo) cuando haces manual capture,
# activa SAVE_IMGS y se crearán carpetas dataset_landmarks/<gesture>_imgs
SAVE_IMGS_ON_MANUAL = False

# -------------------------
# Pedir etiqueta al usuario
# -------------------------
gesture_name = input("🔤 Ingresa el nombre de la seña/letra (ej. A, B, 1): ").strip()
if gesture_name == "":
    print("Debe ingresar un nombre válido. Saliendo.")
    exit(1)

csv_path = os.path.join(DATA_DIR, f"{gesture_name}.csv")
img_dir = os.path.join(DATA_DIR, f"{gesture_name}_imgs")
if SAVE_IMGS_ON_MANUAL:
    os.makedirs(img_dir, exist_ok=True)

# -------------------------
# Contar cuántos ejemplos ya existen (persistente)
# -------------------------
if os.path.exists(csv_path):
    # contamos líneas menos 1 (cabecera)
    with open(csv_path, "r", newline="") as f:
        existing_count = sum(1 for _ in f) - 1
    if existing_count < 0:
        existing_count = 0
else:
    existing_count = 0

session_saves = 0  # contador solo en esta sesión

# -------------------------
# Preparar CSV (escribimos header si no existe)
# -------------------------
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["time", "capture_type"]  # info extra
        # 21 landmarks * 3 coords (x,y,z)
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

# -------------------------
# Inicializar MediaPipe y OpenCV
# -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=max_num_hands,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: no se pudo abrir la cámara.")
    exit(1)

# -------------------------
# Variables de control del temporizador
# -------------------------
last_auto_time = time.time()
img_save_index = existing_count + 1

print(f"Comenzando captura para '{gesture_name}'. Archivo: {csv_path}")
print("Presiona SPACE para capturar manualmente. Presiona 'q' o ESC para salir.")

# -------------------------
# Función para guardar landmarks
# -------------------------
def save_landmarks_to_csv(landmarks, capture_type="manual", frame_bgr=None):
    """Guarda una fila en el CSV: time, capture_type, x0,y0,z0, ..., x20,y20,z20
       Si SAVE_IMGS_ON_MANUAL está activo y frame_bgr provisto, guarda imagen.
    """
    global session_saves, img_save_index

    # Escribir fila
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [time.strftime("%Y-%m-%d %H:%M:%S"), capture_type]
        for lm in landmarks.landmark:
            row += [lm.x, lm.y, lm.z]
        writer.writerow(row)

    # Guardar imagen opcionalmente (solo para capturas manuales)
    if SAVE_IMGS_ON_MANUAL and frame_bgr is not None and capture_type == "manual":
        img_name = os.path.join(img_dir, f"{gesture_name}_{img_save_index:05d}.jpg")
        cv2.imwrite(img_name, frame_bgr)
        img_save_index += 1

    session_saves += 1

# -------------------------
# Loop principal
# -------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: frame no recibido.")
            break

        # Para visualizar texto correctamente, trabajamos con BGR en frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_count = 0
        # Dibujar y procesar landmarks si los hay
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_count += 1
                # Dibujar landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Si corresponde ejecutar captura automática
                current_time = time.time()
                if current_time - last_auto_time >= capture_interval:
                    # (opcional) se puede comprobar confianza desde results.multi_handedness
                    # si está disponible: results.multi_handedness[idx].classification[0].score
                    save_landmarks_to_csv(hand_landmarks, capture_type="auto")
                    last_auto_time = current_time
                    print(f"[AUTO] Guardado ejemplo para '{gesture_name}' (total ahora: {existing_count + session_saves})")

        # -------------------------
        # Mostrar info en pantalla (esquina superior izquierda)
        # -------------------------
        remaining = int(max(0, capture_interval - (time.time() - last_auto_time)))
        total_saved = existing_count + session_saves
        info_1 = f"Next auto: {remaining}s"
        info_2 = f"Saved: {total_saved}"
        # pequeño fondo para legibilidad
        cv2.rectangle(frame, (5,5), (175,60), (0,0,0), -1)  # caja negra semilla
        cv2.putText(frame, info_1, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
        cv2.putText(frame, info_2, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

        # Mostrar conteo de manos detectadas (opcional)
        cv2.putText(frame, f"Hands: {hand_count}", (frame.shape[1]-110,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1, cv2.LINE_AA)

        cv2.imshow("Captura landmarks (SPACE=guardar, q/ESC=salir)", frame)

        # -------------------------
        # Manejo de teclas
        # -------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            if results.multi_hand_landmarks:
                # guardamos la(s) mano(s) detectada(s) en ese frame (por simplicidad, guardamos la primera)
                save_landmarks_to_csv(results.multi_hand_landmarks[0], capture_type="manual", frame_bgr=frame)
                print(f"[MANUAL] Guardado ejemplo para '{gesture_name}' (total ahora: {existing_count + session_saves})")
            else:
                print("No se detectó mano para guardar en este frame.")
        elif key == 27 or key == ord("q"):  # ESC o q
            print("Saliendo...")
            break

except KeyboardInterrupt:
    print("Interrumpido por usuario (Ctrl+C).")

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Sesión finalizada. Ejemplos totales para '{gesture_name}': {existing_count + session_saves}")
    print(f"CSV guardado en: {csv_path}")
    if SAVE_IMGS_ON_MANUAL:
        print(f"Imágenes guardadas en: {img_dir}")
