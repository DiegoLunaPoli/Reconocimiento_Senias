import cv2  # OpenCV para manejar la cámara
import mediapipe as mp  # Librería para detección de manos
import csv  # Para guardar los landmarks en archivos .csv
import os  # Para manejar directorios
import numpy as np  # Para trabajar con vectores/matrices

# Inicializamos MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Carpeta donde se guardarán los landmarks
DATA_DIR = "dataset_landmarks"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Nombre del gesto a capturar (ejemplo: "A", "B", "1", "2")
gesture_name = input("👉 Escribe el nombre de la seña que vas a capturar: ")

# Archivo CSV donde guardaremos las coordenadas
csv_path = os.path.join(DATA_DIR, f"{gesture_name}.csv")
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)

# Encabezados del CSV: x,y,z para cada landmark
headers = []
for i in range(21):  # MediaPipe detecta 21 puntos clave en la mano
    headers += [f"x{i}", f"y{i}", f"z{i}"]
csv_writer.writerow(headers)

# Inicializamos la captura de video
cap = cv2.VideoCapture(0)

# Activamos el modelo de MediaPipe
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # solo detectamos una mano por simplicidad
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    print("📸 Presiona 'q' para salir del programa.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertimos la imagen a RGB (MediaPipe trabaja en RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detectamos la mano
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujamos los puntos clave en la imagen
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extraemos coordenadas x, y, z de cada landmark
                row = []
                for landmark in hand_landmarks.landmark:
                    row += [landmark.x, landmark.y, landmark.z]

                # Guardamos la fila en el CSV
                csv_writer.writerow(row)

        # Mostramos la imagen con landmarks
        cv2.imshow("Captura de Landmarks", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Cerramos todo
cap.release()
cv2.destroyAllWindows()
csv_file.close()
print(f"✅ Datos guardados en {csv_path}")
