
# Proyecto de Reconocimiento de Señas y Emociones

Este proyecto incluye tres aplicaciones basadas en visión por computadora:

1. **Reconocimiento de Señas en Tiempo Real**: Utiliza **Mediapipe** para rastrear las manos y reconocer las letras en lenguaje de señas.
2. **Entrenador de Señas**: Captura las posiciones de las manos y guarda los puntos clave en archivos CSV, etiquetados con las señas correspondientes.
3. **Reconocimiento de Emociones con YOLOv8**: Utiliza un modelo de **YOLOv8** para detectar rostros y emociones en tiempo real a través de la cámara web.

## Descripción

### 1. **Lector de Señas**
El lector de señas reconoce letras del alfabeto de señas en tiempo real utilizando **Mediapipe** para el rastreo de las manos. Los modelos de letras están entrenados y almacenados en archivos CSV. La detección de letras se realiza comparando las coordenadas de los puntos clave de las manos con los modelos preentrenados.

### 2. **Entrenador de Señas**
El entrenador captura datos de las manos usando **Mediapipe** y los guarda en archivos CSV. Estos datos contienen las coordenadas de 21 puntos clave de las manos y están etiquetados con la seña correspondiente. Los archivos CSV generados pueden usarse para entrenar un modelo de reconocimiento de señas.

### 3. **Reconocimiento de Emociones con YOLOv10**
Utiliza un modelo **YOLOv8** preentrenado para detectar rostros y emociones en tiempo real. El modelo puede identificar personas en imágenes y mostrar las emociones detectadas. Los resultados se visualizan en la pantalla con la caja delimitadora alrededor de los rostros y el nombre de la emoción detectada.

## Tecnologías Utilizadas

- **Python 3.12.8**
- **OpenCV**
- **Mediapipe**
- **YOLOv10 (Ultralytics)**
- **NumPy**
- **CSV**

## Instalación

Asegúrate de tener **Python 3.12.8** instalado en tu sistema. Si no lo tienes, puedes descargarlo desde el sitio oficial de [Python](https://www.python.org/downloads/release/python-3128/).

Para instalar las dependencias necesarias, sigue estos pasos:

1. Crea un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate   # En Linux/Mac
   source env\Scripts set activate    # En Windows
   ```

2. Instala las dependencias necesarias a través del archivo `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt**

   ```txt
   opencv-python
   mediapipe
   numpy
   ultralytics
   torch
   ```

## Uso

### Lector de Señas

1. **Inicia el script del lector de señas**:
   ```bash
   python lector_senas.py
   ```
2. **Captura la cámara**: El programa accederá a la cámara web y comenzará a rastrear las manos.
3. **Detección de letras**: Las letras detectadas se mostrarán en la pantalla. Si presionas **espacio**, la letra se añadirá a la palabra en curso.
4. **Salir**: Para salir del programa, presiona la tecla **Esc**.

### Entrenador de Señas

1. **Inicia el script del entrenador de señas**:
   ```bash
   python entrenador_senas.py
   ```
2. **Captura la mano**: Coloca tu mano frente a la cámara.
3. **Guardar datos**: Los puntos clave de las manos se guardarán en un archivo CSV etiquetado con la seña correspondiente.
4. **Salir**: Para finalizar, presiona la tecla **q**.

### Reconocimiento de Emociones

1. **Inicia el script de detección de emociones**:
   ```bash
   python reconocimiento_emociones.py
   ```
2. **Captura la cámara**: El programa iniciará la webcam y comenzará a detectar emociones en los rostros de las personas.
3. **Mostrar resultados**: Se dibujarán cajas alrededor de los rostros con el nombre de la emoción detectada.
4. **Salir**: Para salir del programa, presiona **q**.

## Estructura del Proyecto

```
Proyecto_Python/
├── CSV_Abecedario/
│   ├── [Modelos CSV de letras]
├── lector_senas.py
├── entrenador_senas.py
└── reconocimiento_emociones.py
```

## Posibles Mejoras y Futuro

- **Reconocimiento Facial**: Integrar la detección de emociones con la detección de rostros para mostrar un sistema completo de reconocimiento facial y de emociones.
- **Mejorar el Reconocimiento de Señas**: Mejorar el modelo de reconocimiento de señas con redes neuronales entrenadas con los datos CSV generados.
- **Ampliación de las Emociones Detectadas**: Ampliar el sistema de emociones para detectar más categorías de emociones en los rostros.

## Créditos

- **Mediapipe**: Librería de Google para el rastreo de manos.
- **YOLOv10 (Ultralytics)**: Modelo de detección de objetos, utilizado para la detección de rostros y emociones.

****
