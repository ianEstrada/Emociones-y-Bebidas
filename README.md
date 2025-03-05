
# Proyecto de Reconocimiento de Señas y Emociones

## Menú de Idioma

Selecciona un idioma para leer la documentación:

1. [Español](#proyecto-de-reconocimiento-de-señas-y-emociones)
2. [English](#sign-language-and-emotions-recognition-project)

---

## Proyecto de Reconocimiento de Señas y Emociones

Este proyecto incluye tres aplicaciones basadas en visión por computadora:

- **Reconocimiento de Señas en Tiempo Real**: Utiliza Mediapipe para rastrear las manos y reconocer las letras en lenguaje de señas.
- **Entrenador de Señas**: Captura las posiciones de las manos y guarda los puntos clave en archivos CSV, etiquetados con las señas correspondientes.
- **Reconocimiento de Emociones con FER**: Utiliza el modelo FER (Facial Expression Recognition) para detectar emociones en tiempo real a través de la cámara web.

### Descripción

1. **Lector de Señas**  
El lector de señas reconoce letras del alfabeto de señas en tiempo real utilizando Mediapipe para el rastreo de las manos. Los modelos de letras están entrenados y almacenados en archivos CSV. La detección de letras se realiza comparando las coordenadas de los puntos clave de las manos con los modelos preentrenados.

2. **Entrenador de Señas**  
El entrenador captura datos de las manos usando Mediapipe y los guarda en archivos CSV. Estos datos contienen las coordenadas de 21 puntos clave de las manos y están etiquetados con la seña correspondiente. Los archivos CSV generados pueden usarse para entrenar un modelo de reconocimiento de señas.

3. **Reconocimiento de Emociones con FER**  
Utiliza el modelo **FER** preentrenado para detectar emociones en tiempo real. El modelo puede identificar las emociones de las personas en imágenes y mostrar los resultados con la caja delimitadora alrededor de los rostros y el nombre de la emoción detectada.

### Tecnologías Utilizadas

- Python 3.12.8
- OpenCV
- Mediapipe
- FER (Facial Expression Recognition)
- NumPy
- CSV

### Instalación

Asegúrate de tener Python 3.12.8 instalado en tu sistema. Si no lo tienes, puedes descargarlo desde el sitio oficial de Python.

Para instalar las dependencias necesarias, puedes usar el siguiente comando para instalar todos los paquetes de una vez:


1. Crea un entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
source venv/bin/activate   # En Linux/Mac
source env\Scripts\activate  # En Windows
```

2. Instala las dependencias necesarias a través del archivo:

```bash
pip install opencv-python mediapipe numpy fer
```

### Uso

1. **Lector de Señas**  
Inicia el script del lector de señas:

```bash
python lector_senas.py
```

- Captura la cámara: El programa accederá a la cámara web y comenzará a rastrear las manos.
- Detección de letras: Las letras detectadas se mostrarán en la pantalla. Si presionas espacio, la letra se añadirá a la palabra en curso.
- Salir: Para salir del programa, presiona la tecla Esc.

2. **Entrenador de Señas**  
Inicia el script del entrenador de señas:

```bash
python entrenador_senas.py
```

- Captura la mano: Coloca tu mano frente a la cámara.
- Guardar datos: Los puntos clave de las manos se guardarán en un archivo CSV etiquetado con la seña correspondiente.
- Salir: Para finalizar, presiona la tecla q.

3. **Reconocimiento de Emociones**  
Inicia el script de detección de emociones:

```bash
python reconocimiento_emociones.py
```

- Captura la cámara: El programa iniciará la webcam y comenzará a detectar emociones en los rostros de las personas.
- Mostrar resultados: Se dibujarán cajas alrededor de los rostros con el nombre de la emoción detectada.
- Salir: Para salir del programa, presiona q.

### Estructura del Proyecto

```
Proyecto_Python/
├── CSV_Abecedario/
│   ├── [Modelos CSV de letras]
├── lector_senas.py
├── entrenador_senas.py
└── reconocimiento_emociones.py
```

### Posibles Mejoras y Futuro

- **Reconocimiento Facial**: Integrar la detección de emociones con la detección de rostros para mostrar un sistema completo de reconocimiento facial y de emociones.
- **Mejorar el Reconocimiento de Señas**: Mejorar el modelo de reconocimiento de señas con redes neuronales entrenadas con los datos CSV generados.
- **Ampliación de las Emociones Detectadas**: Ampliar el sistema de emociones para detectar más categorías de emociones en los rostros.

---

# Sign Language and Emotions Recognition Project

## Language Menu

Select a language to read the documentation:

1. [Español](#proyecto-de-reconocimiento-de-señas-y-emociones)
2. [English](#sign-language-and-emotions-recognition-project)

---

## Sign Language and Emotions Recognition Project

This project includes three applications based on computer vision:

- **Real-time Sign Language Recognition**: Uses Mediapipe to track hands and recognize letters in sign language.
- **Sign Language Trainer**: Captures hand positions and saves key points in CSV files, labeled with corresponding signs.
- **Emotion Recognition with FER**: Uses the FER (Facial Expression Recognition) model to detect emotions in real-time through the webcam.

### Description

1. **Sign Language Reader**  
The sign language reader recognizes letters from the sign language alphabet in real-time using Mediapipe for hand tracking. The sign models are trained and stored in CSV files. Letter detection is done by comparing the coordinates of key hand points with pre-trained models.

2. **Sign Language Trainer**  
The trainer captures hand data using Mediapipe and saves it in CSV files. This data contains the coordinates of 21 key hand points and is labeled with the corresponding sign. The generated CSV files can be used to train a sign language recognition model.

3. **Emotion Recognition with FER**  
Uses the pre-trained **FER** model to detect emotions in real-time. The model can identify emotions of people in images and display the results with a bounding box around the faces and the name of the detected emotion.

### Technologies Used

- Python 3.12.8
- OpenCV
- Mediapipe
- FER (Facial Expression Recognition)
- NumPy
- CSV

### Installation

Make sure you have Python 3.12.8 installed on your system. If not, you can download it from the official Python website.

To install the necessary dependencies, you can use the following command to install all packages at once:

1. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
source env\Scripts\activate  # On Windows
```
2. Install the necessary dependencies through the following command:

```bash
pip install opencv-python mediapipe numpy fer
```

### Usage

1. **Sign Language Reader**  
Run the sign language reader script:

```bash
python lector_senas.py
```

- Camera Capture: The program will access the webcam and start tracking the hands.
- Letter Detection: Detected letters will appear on the screen. If you press space, the letter will be added to the current word.
- Exit: To exit the program, press the Esc key.

2. **Sign Language Trainer**  
Run the sign language trainer script:

```bash
python entrenador_senas.py
```

- Hand Capture: Place your hand in front of the camera.
- Save Data: The key points of the hands will be saved in a CSV file labeled with the corresponding sign.
- Exit: To finish, press the q key.

3. **Emotion Recognition**  
Run the emotion recognition script:

```bash
python reconocimiento_emociones.py
```

- Camera Capture: The program will start the webcam and begin detecting emotions on people's faces.
- Display Results: Bounding boxes will be drawn around faces with the detected emotion name.
- Exit: To exit the program, press the q key.

### Project Structure

```
Proyecto_Python/
├── CSV_Abecedario/
│   ├── [CSV Letter Models]
├── lector_senas.py
├── entrenador_senas.py
└── reconocimiento_emociones.py
```

### Possible Improvements and Future

- **Facial Recognition**: Integrating emotion detection with facial recognition to create a complete facial and emotion recognition system.
- **Improving Sign Language Recognition**: Enhancing the sign language recognition model with neural networks trained with the CSV data.
- **Expanding Detected Emotions**: Expanding the emotion detection system to recognize more categories of emotions in faces.

---

## Credits

- **Mediapipe**: Google library for hand tracking.
- **FER**: Emotion detection model, used for face and emotion detection.

***