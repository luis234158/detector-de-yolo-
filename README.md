# Detector de monedas colombianas con YOLO11

Sistema de visión por computador que detecta, clasifica y cuenta monedas colombianas en tiempo real usando una cámara web y un modelo **YOLO11** entrenado con un dataset propio. Además de reconocer cada moneda, calcula el valor total en pesos.

**Clases:** `50` · `100` · `200` · `500` · `1000`

---

## Estructura

```
detector-monedas-yolo/
├── detector.py           # Detección en tiempo real + suma del dinero
├── capturar.py           # Toma fotos con la cámara para armar el dataset
├── camaras.py            # Lista las cámaras disponibles y su índice
├── entrenamiento.ipynb   # Notebook de Google Colab para entrenar el modelo
├── requirements.txt
└── best.pt               # Modelo entrenado (pesos)
```

---

## Flujo de trabajo

```
capturar.py  →  Roboflow (etiquetado)  →  Colab (entrenamiento)  →  best.pt  →  detector.py
```

### 1. Captura de imágenes

Con la cámara fija desde arriba, sobre un fondo liso:

```bash
python capturar.py
```

`ESPACIO` guarda una foto y `q` sale. Las imágenes quedan en `fotos/`.

Recomendaciones para el dataset:
- 5–8 monedas mezcladas por foto, mostrando cara y sello.
- Variar posición, iluminación y monedas tocándose.
- Mínimo **50 etiquetas por denominación**.
- Incluir 15–20 fotos **sin monedas** (mesa vacía, mano, papeles) marcadas como *Null*, para reducir falsos positivos.

### 2. Etiquetado en Roboflow

1. Subir la carpeta `fotos/` al proyecto (*Object Detection*).
2. Etiquetar manualmente cada moneda con su clase.
3. Generar una versión:
   - Resize **640×640** con **Fit (black edges)**. No usar *Stretch*, porque deforma las monedas.
   - Aumentaciones: rotación ±15°, brillo ±20 %, flip horizontal.
   - División 70/20/10 (train/valid/test).

### 3. Entrenamiento en Google Colab

Abrir `entrenamiento.ipynb` en Colab con GPU T4, poner la API key de Roboflow y ejecutar las celdas en orden. Al final se descarga `best.pt`.

### 4. Detección en tiempo real

```bash
pip install -r requirements.txt
python camaras.py      # para saber qué índice tiene la cámara
python detector.py
```

`q` para salir.

---

## Configuración de `detector.py`

| Parámetro | Valor | Descripción |
|---|---|---|
| `CAMARA` | `1` | Índice de la cámara (0 = integrada, 1 = USB) |
| `CONF` | `0.6` | Confianza mínima de detección |
| `IOU` | `0.4` | Umbral para eliminar cajas encimadas |
| `RELACION_MIN` / `RELACION_MAX` | `0.75` / `1.33` | Relación ancho/alto permitida (las monedas son casi cuadradas) |
| `AREA_MAX` | `0.08` | Fracción máxima de la imagen que puede ocupar una moneda |

Además del modelo, el detector aplica un **filtro geométrico** que descarta cajas alargadas o demasiado grandes, y usa **NMS agnóstico** para evitar detecciones duplicadas sobre la misma moneda.

---

## Lecciones aprendidas

| Problema observado | Causa | Solución |
|---|---|---|
| Confundía la ropa blanca con monedas de 500 | El modelo nunca vio imágenes sin monedas | Agregar imágenes de fondo (*Null*) + filtro geométrico |
| Clasificaba casi todo como 500 o 1000 | Solo 37 monedas etiquetadas en la versión 1 (4–9 por clase) | Aumentar a 50+ etiquetas por clase |
| Mal desempeño con la webcam | Fotos de entrenamiento tomadas con otra cámara | Capturar el dataset con la misma cámara y montaje |
| Monedas deformadas en entrenamiento | Preprocesamiento *Stretch* | Usar *Fit (black edges)* |
| Métricas poco confiables | Validación con solo 7 monedas | Dataset más grande y balanceado |

Las monedas de 50, 100 y 200 se distinguen principalmente por tamaño y color, y YOLO solo percibe el tamaño en píxeles. Por eso es clave mantener la **cámara a altura fija**.

---

## Tecnologías

- [Ultralytics YOLO11](https://docs.ultralytics.com/)
- [OpenCV](https://opencv.org/)
- [Roboflow](https://roboflow.com/) – etiquetado y gestión del dataset
- [Google Colab](https://colab.research.google.com/) – entrenamiento con GPU

---

**Autor:** Luis Ruiz – Ingeniería Mecatrónica, Universidad Militar Nueva Granada
