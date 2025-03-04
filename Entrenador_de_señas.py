import cv2
import mediapipe as mp
import os
import time

# Inicializar MediaPipe para el reconocimiento de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Configuración de directorios
image_dir = "imagenes"
label_dir = "etiquetas"
os.makedirs(image_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# Función para obtener las coordenadas de las manos y guardarlas en el formato YOLO
def obtener_coordenadas_yolo(hand_landmarks, image_width, image_height):
    annotations = []
    for hand in hand_landmarks:
        # Obtener las coordenadas de los puntos clave de la mano
        for landmark in hand.landmark:
            # Normalizar las coordenadas (en el rango [0, 1])
            x = landmark.x * image_width
            y = landmark.y * image_height
            annotations.append((x, y))
    
    # Cálculo de la caja delimitadora
    if len(annotations) > 0:
        xmin = min(annotations, key=lambda x: x[0])[0]
        ymin = min(annotations, key=lambda x: x[1])[1]
        xmax = max(annotations, key=lambda x: x[0])[0]
        ymax = max(annotations, key=lambda x: x[1])[1]
        
        # Normalizar la caja delimitadora
        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        
        return x_center, y_center, width, height
    return None

# Iniciar la cámara
cap = cv2.VideoCapture(0)

# Variable para el nombre de la imagen
counter = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Obtener el tamaño de la imagen
    image_height, image_width, _ = frame.shape
    
    # Convertir la imagen a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # Dibujar las detecciones
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Si se presiona la tecla espacio, capturar imagen y guardar archivo
        if cv2.waitKey(1) & 0xFF == ord(' '):
            # Pedir al usuario el nombre del archivo
            label_name = input("Introduce el nombre de la letra (ejemplo: A): ")
            
            # Guardar la imagen
            img_filename = f"{label_name}_{counter}.png"
            img_filepath = os.path.join(image_dir, img_filename)
            cv2.imwrite(img_filepath, frame)
            
            # Obtener las coordenadas de las manos
            coordenadas = obtener_coordenadas_yolo(results.multi_hand_landmarks, image_width, image_height)
            
            if coordenadas:
                # Guardar las coordenadas en un archivo de texto en formato YOLO
                txt_filename = f"{label_name}_{counter}.txt"
                txt_filepath = os.path.join(label_dir, txt_filename)
                with open(txt_filepath, 'w') as f:
                    f.write(f"0 {coordenadas[0]} {coordenadas[1]} {coordenadas[2]} {coordenadas[3]}\n")
            
            # Incrementar el contador para la próxima imagen
            counter += 1

    # Mostrar la imagen con las manos detectadas
    cv2.imshow("Detector de Señales de Manos", frame)

    # Salir si presionas 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
