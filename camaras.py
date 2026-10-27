"""
Lista las cámaras disponibles y muestra una vista previa de cada una.

Sirve para saber qué número poner en CAMARA dentro de detector.py y capturar.py.

Uso:
    python camaras.py
"""

import cv2


def main():
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok, frame = cap.read()
        if ok:
            print(f"Cámara {i}: funciona")
            cv2.imshow(f"Camara {i}", frame)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()
        else:
            print(f"Cámara {i}: no disponible")
        cap.release()


if __name__ == "__main__":
    main()
