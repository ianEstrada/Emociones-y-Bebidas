import cv2
from fer import FER
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageOps

# --- CONFIGURACIÓN INICIAL ---
emotion_detector = FER(mtcnn=True) 

EMOTION_COLORS = {
    'happy': (0, 255, 0),
    'sad': (255, 100, 0),      
    'angry': (0, 0, 255),      
    'surprise': (0, 255, 255), 
    'fear': (128, 0, 128),     
    'neutral': (200, 200, 200),
    'disgust': (0, 128, 0),    
}
DEFAULT_COLOR = (120, 120, 120) 

modo_app = "webcam"
frame_count = 0
PROCESAR_CADA_N_FRAMES = 1

# --- LÓGICA DE DETECCIÓN REUTILIZABLE ---
def procesar_imagen(frame):
    all_faces = emotion_detector.detect_emotions(frame)
    
    if not all_faces:
        status_label.config(text="Buscando rostro...")
        return frame

    for face_data in all_faces:
        box = face_data['box']
        emotions = face_data['emotions']
        (x, y, w, h) = box
        
        top_emotion = max(emotions, key=emotions.get)
        color = EMOTION_COLORS.get(top_emotion, DEFAULT_COLOR)
        
        status_label.config(text=f"Emoción detectada: {top_emotion.capitalize()}")
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        emotion_text = top_emotion.capitalize()
        (text_w, text_h), _ = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(frame, (x, y - text_h - 15), (x + text_w + 10, y - 5), color, -1)
        cv2.putText(frame, emotion_text, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
    return frame

# --- MODIFICACIÓN 2: Lógica de redimensionamiento de imagen ---
def mostrar_frame_en_label(frame, label):
    """Convierte un frame de OpenCV, lo redimensiona y lo muestra en una etiqueta."""
    
    # Obtenemos el tamaño del widget de la etiqueta que contiene el video
    label_w = label.winfo_width()
    label_h = label.winfo_height()

    # Si la ventana se acaba de abrir, sus dimensiones pueden ser 1, lo ignoramos.
    if label_w <= 1 or label_h <= 1:
        return # Evitamos redimensionar a un tamaño inválido

    # Convertimos el frame de OpenCV a una imagen de PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    
    # Redimensionamos la imagen para que se ajuste al tamaño de la etiqueta
    # Image.Resampling.LANCZOS es un algoritmo de alta calidad para redimensionar
    img_resized = ImageOps.pad(img, (label_w, label_h), color='black', centering=(0.5, 0.5))

    imgtk = ImageTk.PhotoImage(image=img_resized)
    
    label.imgtk = imgtk # type: ignore
    label.configure(image=imgtk)


# --- FUNCIONES DE CONTROL DE LA APP (Sin cambios) ---
def cargar_y_analizar_imagen():
    global modo_app
    modo_app = "image"

    filepath = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*")]
    )
    if not filepath:
        reiniciar_camara()
        return

    frame = cv2.imread(filepath)
    if frame is None:
        status_label.config(text="Error: No se pudo cargar la imagen")
        reiniciar_camara()
        return

    frame_procesado = procesar_imagen(frame)
    mostrar_frame_en_label(frame_procesado, video_label)

def reiniciar_camara():
    global modo_app
    if modo_app == "webcam":
        return
    modo_app = "webcam"
    update_frame()

# --- FUNCIÓN DEL BUCLE DE LA WEBCAM (Sin cambios) ---
def update_frame():
    global frame_count
    if modo_app != "webcam":
        return

    ret, frame = cap.read()
    if not ret:
        root.destroy()
        return

    frame = cv2.flip(frame, 1)
    
    frame_count += 1
    frame_procesado = frame
    
    if frame_count % PROCESAR_CADA_N_FRAMES == 0:
        frame_procesado = procesar_imagen(frame)
    else:
        pass 
    
    mostrar_frame_en_label(frame_procesado, video_label)
    
    video_label.after(15, update_frame)

# --- INTERFAZ GRÁFICA ---
root = tk.Tk()
root.title("Detector de Emociones con MTCNN")
root.geometry("1600x900")
root.configure(bg="#2c3e50")

main_title = ttk.Label(root, text="Detector de Emociones", font=("Helvetica", 24, "bold"), foreground="white", background="#2c3e50")
main_title.pack(pady=10)

frame_botones = ttk.Frame(root)
frame_botones.pack(pady=10)

style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)

btn_cargar_imagen = ttk.Button(frame_botones, text="Cargar Imagen", command=cargar_y_analizar_imagen)
btn_cargar_imagen.pack(side="left", padx=10)

btn_iniciar_camara = ttk.Button(frame_botones, text="Iniciar Cámara", command=reiniciar_camara)
btn_iniciar_camara.pack(side="left", padx=10)

video_label = ttk.Label(root)
# --- MODIFICACIÓN 1: Añadimos fill="both" para que la etiqueta ocupe el espacio ---
video_label.pack(pady=10, fill="both", expand=True)

status_label = ttk.Label(root, text="Estado: Buscando...", font=("Helvetica", 14), foreground="white", background="#2c3e50")
status_label.pack(pady=10)

# --- INICIO Y LIMPIEZA ---
cap = cv2.VideoCapture(0)

if cap.isOpened():
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
    cap.set(cv2.CAP_PROP_FPS, 60)
    update_frame()
else: 
    status_label.config(text="Error: No se pudo abrir la cámara.")
    
root.mainloop()

print("Cerrando aplicación...")
cap.release()
