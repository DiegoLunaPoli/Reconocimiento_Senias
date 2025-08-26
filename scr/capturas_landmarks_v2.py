import cv2
import mediapipe as mp
import time
import csv
import os

# ==============================
# 1. Inicializar MediaPipe
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Creamos el detector de manos
hands = mp_hands.Hands(
    static_image_mode=False,     # Detecta en video (True sería solo imágenes)
    max_num_hands=1,             # Una mano por ahora
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# 2. Configuración de captura y dataset
# ==============================
cap = cv2.VideoCapture(0)  # Cámara principal
output_file = "landmarks_dataset.csv"

# Crear archivo si no existe y escribir encabezados
if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        headers = ["time", "capture_type"]  # Info extra: tiempo y tipo de captura
        for i in range(21):  # 21 puntos de la mano
            headers += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(headers)

# ==============================
# 3. Variables para control de tiempo
# ==============================
last_capture_time = time.time()
capture_interval = 5  # segundos para captura automática

# ==============================
# 4. Función para guardar landmarks
# ==============================
def save_landmarks(landmarks, capture_type="manual"):
    """Guarda coordenadas de landmarks en un CSV"""
    with open(output_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        row = [time.strftime("%H:%M:%S"), capture_type]
        for lm in landmarks.landmark:
            row += [lm.x, lm.y, lm.z]
        writer.writerow(row)

# ==============================
# 5. Loop principal
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Si detecta manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibujar puntos en la mano
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --------------- Captura automática cada X segundos ---------------
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                save_landmarks(hand_landmarks, capture_type="auto")
                last_capture_time = current_time
                print("✅ Captura automática realizada")

    # --------------- Mostrar reloj en pantalla ---------------
    elapsed_time = int(time.time() - last_capture_time)
    remaining = capture_interval - elapsed_time
    if remaining < 0:  # Evita que quede en negativo
        remaining = 0
    cv2.putText(frame, f"Next auto: {remaining}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --------------- Mostrar video en ventana ---------------
    cv2.imshow("Captura de landmarks", frame)

    # --------------- Captura manual con SPACE ---------------
    key = cv2.waitKey(1) & 0xFF
    if key == 32:  # Tecla SPACE
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                save_landmarks(hand_landmarks, capture_type="manual")
                print("✋ Captura manual realizada")
    elif key == 27:  # ESC para salir
        break

# ==============================
# 6. Liberar recursos
# ==============================
cap.release()
cv2.destroyAllWindows()
