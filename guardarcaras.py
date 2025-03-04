from ultralytics import YOLO
import torch

def main():
    # Verificar si hay GPU disponible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Cargar el modelo YOLO preentrenado
    model = YOLO("yolo11n.pt")  # Asegúrate de que el nombre del modelo sea correcto
    model.to(device)  # Enviar modelo a la GPU (si está disponible)
    
    # Ruta al archivo de configuración del dataset
    data_path = "C:\\Users\\Lightning\\Documents\\Proyecto_Python\\data.yaml"
    
    # Entrenar el modelo
    results = model.train(data=data_path, epochs=500, imgsz=640, device=device)
    
    # Guardar el modelo entrenado
    model.save("best_model.pt")
    print("Entrenamiento completado y modelo guardado como 'best_model.pt'")

if __name__ == "__main__":
    main()