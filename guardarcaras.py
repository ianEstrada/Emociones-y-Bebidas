from ultralytics import YOLO
import torch
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)



 # Enviar modelo a la GPU
    model.to(device)

# Train the model with MPS
    results = model.train(data="C:\\Users\\Lightning\\Documents\\Proyecto_Python\\data.yaml", epochs=30, imgsz=640, device=device)

