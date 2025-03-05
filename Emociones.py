import cv2
from fer import FER
import os
import sys

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Inicializar detector de emociones con el modelo cargado
emotion_detector = FER()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Parámetros configurables
resolution_width = 1280  # Ancho de la resolución
resolution_height = 720  # Alto de la resolución
fps = 60  # Tasa de fotogramas por segundo


# Configurar la resolución y la tasa de fotogramas
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
cap.set(cv2.CAP_PROP_FPS, fps)

# Crear ventana y ajustarla automáticamente según la resolución de la cámara
cv2.namedWindow("Detección de Emociones", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detección de Emociones", resolution_width, resolution_height)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB (OpenCV usa BGR por defecto)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convertir a escala de grises para detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame_rgb[y:y+h, x:x+w]  # Usar RGB para la detección de emociones
        emotions, _ = emotion_detector.top_emotion(face)

        if emotions:
            cv2.putText(frame, f"Emocion: {emotions}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar el video con detección de emociones
    cv2.imshow("Detección de Emociones", frame)

    # Salir cuando se presione la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
