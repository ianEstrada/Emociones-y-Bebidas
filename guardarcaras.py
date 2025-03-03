import torch
from ultralytics import YOLO
import os

def train_model():
    # Verifica si la GPU está disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Dispositivo usado:", device)

    # Ruta del modelo entrenado
    model_path = 'yolo11n_trained.pt'  # Archivo de salida del modelo entrenado

    # Verifica si el modelo ya ha sido entrenado (es decir, si el archivo existe)
    if os.path.exists(model_path):
        print(f"El modelo ya está entrenado y guardado en: {model_path}")
        return  # Sale de la función si el modelo ya está entrenado

    # Cargar modelo YOLO
    model = YOLO('yolo11n.pt')  # Usa un modelo más ligero para pruebas

    # Enviar modelo a la GPU
    model.to(device)

    # Entrenar
    print("Entrenando el modelo...")
    model.train(
        data='C:/Users/Lightning/Documents/Proyecto_Python/data.yaml',
        epochs=30,
        imgsz=640,
        device=device
    )

    # Guardar el modelo entrenado
    model.save(model_path)
    print(f"Modelo entrenado guardado en: {model_path}")

if __name__ == '__main__':
    train_model()
