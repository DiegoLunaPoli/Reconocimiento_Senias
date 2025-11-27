import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json

# ---------------------------------------
# 1. CARGAR MODELO Y ETIQUETAS
# ---------------------------------------

modelo = tf.keras.models.load_model(
    "C:/Users/julia/OneDrive PolitecnicoGrancolombombiano/Documentos/U/SEMESTRE 6/SISTEMAS OPERACIONALES/PROG/proyecto/Reconocimiento_Senias/modelo/modelo_signos.h5"
)

with open("dataset_landmarks_limpios/label2id.json", "r", encoding="utf-8") as f:
    label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}

# ---------------------------------------
# 2. CONFIGURAR MEDIAPIPE (API NUEVA)
# ---------------------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


# ---------------------------------------
# 3. INICIAR CAMARA
# ---------------------------------------

cap = cv2.VideoCapture(0)
print("Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    landmarks_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for l in hand_landmarks.landmark:
                landmarks_list.extend([l.x, l.y, l.z])

        entrada = np.array(landmarks_list).reshape(1, -1)

        pred = modelo.predict(entrada, verbose=0)
        index_pred = np.argmax(pred)
        letra = id2label[index_pred]

        cv2.putText(frame, f"Letra: {letra}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    else:
        cv2.putText(frame, "No se detecta mano", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Reconocimiento de Letras", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
