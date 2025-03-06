import os
import cv2
from fer import FER

# Configura TensorFlow y OpenCV
cap = cv2.VideoCapture(0)  # Abre la cámara
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

# Configura la resolución y la tasa de fotogramas
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

# Inicializa el detector de emociones y el clasificador de rostros
emotion_detector = FER()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Bucle principal para procesar cada fotograma
while True:
    ret, frame = cap.read()  # Captura un fotograma
    if not ret:
        print("Error: No se pudo capturar el frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convierte a escala de grises
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)  # Detecta rostros

    for (x, y, w, h) in faces:
        face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)  # Extrae el rostro
        emotions, _ = emotion_detector.top_emotion(face_rgb)  # Detecta emociones

        if emotions:
            cv2.putText(frame, f"Emocion: {emotions}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Muestra emoción
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Dibuja rectángulo alrededor del rostro

    cv2.imshow("Detección de Emociones", frame)  # Muestra el video procesado

    if cv2.waitKey(1) & 0xFF == 27:  # Si se presiona 'Esc', sale del bucle
        break

cap.release()  # Libera la cámara
cv2.destroyAllWindows()  # Cierra todas las ventanas de OpenCV
