import cv2
import torch
from ultralytics import YOLO
import numpy as np

# Cargar el modelo YOLOv8 entrenado
model = YOLO("C:\\Users\\Lightning\\Documents\\Proyecto_Python\\best_model.pt")

# Iniciar la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_frame = cv2.merge([gray_frame, gray_frame, gray_frame])

    # Realizar la detección con YOLOv8
    results = model.predict(source="0", show=True, conf=0.2)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Liberar recursos
cap.release()
cv2.destroyAllWindows()
