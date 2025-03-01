import torch
from ultralytics import YOLO

def train_model():
    # Verifica si la GPU está disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Dispositivo usado:", device)

    # Cargar modelo YOLO
    model = YOLO('yolov10b.pt')  # Usa un modelo más ligero para pruebas

    # Enviar modelo a la GPU
    model.to(device)

    # Entrenar
    model.train(
        data='C:/Users/Lightning/Documents/Proyecto_Python/data.yaml',
        epochs=30,
        imgsz=640,
        batch=8,  # Baja el batch si sigue fallando
        device=device
    )

if __name__ == '__main__':
    train_model()
