import os
import sys
import cv2
from fer import FER

# Función para obtener la ruta correcta dependiendo de si estamos empaquetados o no
def resource_path(relative_path):
    try:
        # Si estamos empaquetados en un .exe, usa sys._MEIPASS
        if getattr(sys, 'frozen', False):
            # Cuando estamos empaquetados, los archivos se extraen en un directorio temporal
            base_path = sys._MEIPASS
        else:
            # Si no estamos empaquetados, usamos la ruta relativa actual
            base_path = os.path.abspath(".")
        
        return os.path.join(base_path, relative_path)
    except Exception as e:
        print("Error al obtener la ruta del recurso:", e)
        return None

# Rutas para el archivo XML
xml_path = resource_path('Data Models/haarcascade_frontalface_default.xml')

# Verificación de ruta del archivo XML
if not xml_path:
    print("Error: No se pudo encontrar el archivo XML.")
    sys.exit(1)

# Cargar el clasificador de caras
face_cascade = cv2.CascadeClassifier(xml_path)

# Crear un detector de emociones (utilizando el modelo preentrenado por defecto)
emotion_detector = FER()

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

cv2.namedWindow("Detección de Emociones", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detección de Emociones", 1280,720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar caras
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Detectar emociones en la cara detectada
        emotions, _ = emotion_detector.top_emotion(face)

        # Si se detecta una emoción, mostrarla en la pantalla
        if emotions:
            cv2.putText(frame, f"Emocion: {emotions}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar el cuadro con las emociones
    cv2.imshow("Detección de Emociones", frame)

    # Salir cuando se presione la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
