import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Captura de video
cap = cv2.VideoCapture(0)

def guardar_en_csv(lecturas, nombre_archivo):
    with open(nombre_archivo, mode='a', newline='') as archivo:
        writer = csv.writer(archivo)
        writer.writerow(lecturas)

# Ruta completa para guardar el archivo
ruta_archivo = r'C:\Users\Lightning\Documents\Proyecto_Python\CSV_Abecedario\FUCKYOU_Models.csv'

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer coordenadas de los 21 puntos clave
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])  # Guardamos x, y, z

            # Convertir a numpy array para procesarlo mejor
            landmarks = np.array(landmarks).flatten() 

            # Escribimosla etiqueta para esta serie de datos
            etiqueta = 'FY'

            # Crear la lista para guardar en CSV, con la etiqueta primero
            lectura = [etiqueta] + list(landmarks)
            print(lectura)

            # Guardar los datos en el CSV
            guardar_en_csv(lectura, ruta_archivo)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar Recursos
cap.release()
cv2.destroyAllWindows()
