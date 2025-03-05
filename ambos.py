import cv2
from fer import FER
import mediapipe as mp
import numpy as np
import os

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

# Inicializar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Cargar modelos de letras
ruta_modelos = r'C:\Users\Lightning\Documents\Proyecto_Python\CSV_Abecedario'

def cargar_modelos():
    modelos = {}
    for archivo in os.listdir(ruta_modelos):
        if archivo.endswith(".csv"):
            letra = archivo.split('_')[0]  
            ruta_archivo = os.path.join(ruta_modelos, archivo)
            modelos[letra] = np.loadtxt(ruta_archivo, delimiter=',', usecols=range(1, 64))  
    return modelos

modelos = cargar_modelos()

def reconocer_letra(landmarks, modelos, umbral=0.2):
    letra_reconocida, distancia_minima = None, float('inf')
    for letra, modelo in modelos.items():
        distancia = np.min(np.linalg.norm(modelo - landmarks, axis=1))
        if distancia < distancia_minima:
            distancia_minima, letra_reconocida = distancia, letra

    return letra_reconocida if distancia_minima < umbral else None

# Variables de letra y palabra
palabra = ""  
letra_detectada = None  # Letra reconocida en tiempo real

# Crear ventana para que se ajuste automáticamente
cv2.namedWindow("Detección de Emociones y Letras", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detección de Emociones y Letras", resolution_width, resolution_height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a RGB para la detección de emociones
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convertir a escala de grises para detección de rostros
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Detectar emociones en los rostros
    for (x, y, w, h) in faces:
        face = frame_rgb[y:y+h, x:x+w]  # Usar RGB para la detección de emociones
        emotions, _ = emotion_detector.top_emotion(face)

        if emotions:
            # Cambiar el color a un verde brillante para la emoción
            cv2.putText(frame, f"Emocion: {emotions}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Dibujar un rectángulo alrededor del rostro en color azul
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Procesar las manos para detectar letras
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            letra_detectada = reconocer_letra(landmarks, modelos)

    # Mostrar la palabra y la letra detectada
    # Modificar las coordenadas de la palabra y la letra para que estén más abajo a la izquierda
    cv2.putText(frame, f'Palabra: {palabra}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Letra detectada: {letra_detectada or ""}', (50, resolution_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)

    # Si presiona espacio, agrega la letra detectada a la palabra
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' ') and letra_detectada:
        palabra += letra_detectada

    # Salir con ESC
    if key == 27:
        break

    # Mostrar el video
    cv2.imshow("Detección de Emociones y Letras", frame)

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
