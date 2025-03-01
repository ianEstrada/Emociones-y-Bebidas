import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Cargar el modelo YOLOv8 entrenado
model = YOLO("C:\\Users\\Lightning\\Documents\\Proyecto_Python\\runs\detect\\train4\weights\\best.pt")

# Iniciar la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_frame = cv2.merge([gray_frame, gray_frame, gray_frame])

    # Realizar la detección con YOLOv8
    results = model.predict(gray_frame, conf=0.5)  

    # Dibujar los resultados en la imagen original (frame a color)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = box.conf[0].item()  
            label = result.names[int(box.cls[0])]  

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar la imagen con detecciones
    cv2.imshow("Detección de Emociones con YOLOv8", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
