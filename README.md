
# Proyecto de Detección de Emociones en Tiempo Real

Este proyecto utiliza OpenCV y FER (Facial Expression Recognition) para detectar emociones en tiempo real a través de una cámara web. El sistema identifica los rostros en un video y analiza las emociones predominantes de las personas detectadas.

## Requisitos

- Python 3.x
- OpenCV
- FER (Facial Expression Recognition)

## Estructura del Proyecto

```
Proyecto_Python/
│
└── emotion_recognition.py        # Script principal para la detección de emociones
```

## Uso

1. Clona el repositorio o copia los archivos a tu máquina.
2. Instala las dependencias con:

    ```bash
    pip install tensorflow
    pip install setuptools
    pip install fer
    pip uninstall moviepy
    pip install moviepy==1.0.3
    pip install opencv-python
    pip install mediapipe
    pip install ultralytics
    pip install torch
    ```

3. Ejecuta el script para detectar emociones en tiempo real:

    ```bash
    python emotion_recognition.py
    ```

El modelo usará la cámara web para capturar imágenes y detectar las emociones de los rostros.

## Explicación del Script

1. **Captura de Video**: El script abre la cámara web y ajusta la resolución a 1280x720 y la tasa de fotogramas a 60 FPS.
2. **Detección de Rostros**: Utiliza un clasificador Haar para detectar rostros en cada fotograma.
3. **Reconocimiento de Emociones**: El detector de emociones `FER` analiza cada rostro y predice la emoción más probable (por ejemplo, felicidad, tristeza, enojo).
4. **Visualización**: Dibuja un rectángulo alrededor de cada rostro detectado y muestra la emoción sobre la cabeza de la persona.

---

# Proyecto de Reconocimiento de Bebidas con YOLOv11

Este proyecto utiliza un modelo entrenado con YOLOv11 para detectar y clasificar bebidas en tiempo real usando una cámara web.

## Requisitos

- Python 3.x
- OpenCV
- ultralytics

## Estructura del Proyecto

```
Proyecto_Python/
│
├── best_model.pt                # Modelo entrenado de YOLO11
├── Bebida.py                    # Script principal para la detección en tiempo real
|── EntrenarModelo.py            # Este Script Entrena el modelo yolo11n.pt (genera best_model.pt)
|── yolo11n.pt                   # Modelo de YOLO usado para reconocimiento de objetos
└── dataset/                      # Conjunto de datos de entrenamiento (si se tiene)
```

## Dataset

El conjunto de datos utilizado para entrenar el modelo se puede obtener de [Roboflow - Soft Drinks](https://universe.roboflow.com/test01-fr735/soft-drinks-632ij/dataset/1). Este dataset contiene imágenes etiquetadas de diferentes bebidas, como Coca-Cola, Fanta, Pepsi, Pibb, Root Beer y Sprite.

## Entrenamiento del Modelo

El modelo YOLOv11 fue entrenado con el conjunto de datos mencionado anteriormente. El proceso de entrenamiento incluyó:

1. **Recolección y etiquetado de imágenes**: Utilizando las imágenes de Roboflow.
2. **Organización de datos**: Las imágenes fueron organizadas en carpetas de `train`, `valid` y `test`.
3. **Entrenamiento**: El modelo fue entrenado con el comando:

    ```bash
    python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov11n.pt --cache
    ```

4. **Modelo final**: El mejor modelo entrenado se guarda como `best_model.pt`.

## Uso

1. Clona el repositorio o copia los archivos a tu máquina.
2. Instala las dependencias con:

    ```bash
    pip install tensorflow
    pip install setuptools
    pip install fer
    pip uninstall moviepy
    pip install moviepy==1.0.3
    pip install opencv-python
    pip install mediapipe
    pip install ultralytics
    pip install torch
    ```

3. Ejecuta el script para detectar bebidas en tiempo real:

    ```bash
    python Bebida.py
    ```

El modelo usará la cámara web para capturar imágenes y detectar las bebidas.

---


