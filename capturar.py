"""
Captura de imágenes para armar el dataset.

Toma fotos con la misma cámara y montaje que se usarán en la detección,
para que el modelo entrene con imágenes parecidas a las reales.

Uso:
    python capturar.py

Teclas:
    ESPACIO -> guardar foto
    q       -> salir

Las fotos se guardan en la carpeta fotos/
"""

import os
import cv2

CAMARA = 1          # mismo índice que en detector.py
CARPETA = "fotos"


def main():
    os.makedirs(CARPETA, exist_ok=True)
    cap = cv2.VideoCapture(CAMARA, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {CAMARA}. Prueba otro índice con camaras.py")
        return

    n = len(os.listdir(CARPETA))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vista = frame.copy()
        cv2.putText(vista, f"Fotos: {n}  [ESPACIO]=foto  [q]=salir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Captura", vista)

        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            ruta = os.path.join(CARPETA, f"img_{n:03d}.jpg")
            cv2.imwrite(ruta, frame)
            print(f"Guardada: {ruta}")
            n += 1
        elif k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
