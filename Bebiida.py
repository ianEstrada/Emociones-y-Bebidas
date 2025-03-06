from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
model = YOLO('C:\\Users\\Lightning\\Documents\\Proyecto_Python\\best_model.pt')  # Reemplaza con la ruta de tu modelo entrenado

# Configurar la cámara
cap = cv2.VideoCapture(0)  # 0 es el índice de la cámara por defecto

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error al capturar la imagen.")
        break

    # Realizar la inferencia
    results = model(frame)

    # Dibujar las predicciones en la imagen
    frame = results.render()[0]

    # Mostrar la imagen con las predicciones
    cv2.imshow('Beverage Recognition', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar la ventana de OpenCV
cap.release()
cv2.destroyAllWindows()
