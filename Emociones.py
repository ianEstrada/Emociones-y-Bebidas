import cv2
from fer import FER

cap = cv2.VideoCapture(0)

emotion_detector = FER()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cv2.namedWindow("Detección de Emociones", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detección de Emociones", 800, 600)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
       
        face = frame[y:y+h, x:x+w]

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
