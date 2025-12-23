"""
calibrate.py - Calibración automática de escala píxeles → cm
==============================================================

Uso:
    1. Toma una foto con una regla en la misma distancia que tus plántulas
    2. Ejecuta: python calibrate.py <imagen_con_regla>
    3. Sigue las instrucciones interactivas
"""

import cv2
import sys
from pathlib import Path

def calibrate_scale(image_path):
    """
    Calibración interactiva de escala.
    """
    print("\n" + "="*70)
    print("  CALIBRACIÓN DE ESCALA - Píxeles → Centímetros")
    print("="*70)
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"❌ No se pudo leer: {image_path}")
        return
    
    print("\nInstrucciones:")
    print("  1. Haz clic en el INICIO de una distancia conocida (ej: marca 0cm)")
    print("  2. Haz clic en el FINAL de esa distancia (ej: marca 5cm)")
    print("  3. Presiona ESC para cancelar o cualquier tecla para continuar")
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.line(img_display, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow('Calibración', img_display)
    
    img_display = img.copy()
    cv2.namedWindow('Calibración')
    cv2.setMouseCallback('Calibración', mouse_callback)
    cv2.imshow('Calibración', img_display)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) != 2:
        print("❌ Se necesitan exactamente 2 puntos")
        return
    
    # Calcular distancia en píxeles
    import numpy as np
    distance_px = np.sqrt((points[1][0] - points[0][0])**2 + 
                         (points[1][1] - points[0][1])**2)
    
    print(f"\n✓ Distancia medida: {distance_px:.2f} píxeles")
    
    # Preguntar distancia real
    distance_cm = float(input("\n¿Cuántos centímetros hay entre esos puntos? "))
    
    # Calcular escala
    cm_per_pixel = distance_cm / distance_px
    
    print("\n" + "="*70)
    print("  RESULTADO DE CALIBRACIÓN")
    print("="*70)
    print(f"\nEscala calculada: {cm_per_pixel:.6f} cm/píxel")
    print(f"\nCopia este valor a root_analyzer.py:")
    print(f"  CM_PER_PIXEL = {cm_per_pixel:.6f}")
    print("\n" + "="*70)
    
    # Guardar en archivo
    with open("calibration.txt", "w") as f:
        f.write(f"CM_PER_PIXEL = {cm_per_pixel:.6f}\n")
        f.write(f"# Distancia: {distance_px:.2f} px = {distance_cm} cm\n")
    
    print("\n✓ Guardado en: calibration.txt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nEjemplo:")
        print("  python calibrate.py foto_con_regla.jpg")
        sys.exit(1)
    
    calibrate_scale(sys.argv[1])
