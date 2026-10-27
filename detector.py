"""
Detector de monedas colombianas con YOLO11 (Ultralytics).

Detecta monedas de 50, 100, 200, 500 y 1000 pesos en tiempo real desde una
cámara, las cuenta y suma el valor total.

Uso:
    python detector.py

Teclas:
    q  -> salir
"""

import cv2
from ultralytics import YOLO

# ----------------------------- Configuración -----------------------------
MODELO = "best.pt"        # pesos entrenados (ver entrenamiento.ipynb)
CAMARA = 1                # 0 = cámara del portátil, 1 = cámara USB (usar camaras.py para verificar)
CONF = 0.6                # confianza mínima (bajar si no detecta, subir si hay falsos positivos)
IOU = 0.4                 # umbral de NMS para eliminar cajas encimadas
RELACION_MIN = 0.75       # relación ancho/alto mínima (las monedas son casi cuadradas)
RELACION_MAX = 1.33       # relación ancho/alto máxima
AREA_MAX = 0.08           # fracción máxima de la imagen que puede ocupar una moneda

VALORES = {"50": 50, "100": 100, "200": 200, "500": 500, "1000": 1000}
# -------------------------------------------------------------------------


def es_moneda(x1, y1, x2, y2, w_img, h_img):
    """Filtra detecciones que no tienen forma ni tamaño de moneda."""
    w, h = x2 - x1, y2 - y1
    if h <= 0:
        return False
    relacion = w / h
    area_rel = (w * h) / (w_img * h_img)
    return RELACION_MIN < relacion < RELACION_MAX and area_rel <= AREA_MAX


def main():
    model = YOLO(MODELO)
    cap = cv2.VideoCapture(CAMARA, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {CAMARA}. Prueba otro índice con camaras.py")
        return

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h_img, w_img = frame.shape[:2]

        r = model(frame, conf=CONF, iou=IOU, agnostic_nms=True, verbose=False)[0]

        total = 0
        n = 0
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if not es_moneda(x1, y1, x2, y2, w_img, h_img):
                continue

            clase = model.names[int(box.cls)]
            conf = float(box.conf)
            total += VALORES.get(clase, 0)
            n += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"${clase} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Monedas: {n}  Total: ${total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Detector de monedas", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
