"""
ruler_calibration.py - Calibración de escala mediante regla
===========================================================

Uso:
    python src/ruler_calibration.py <imagen>           # Automático
    python src/ruler_calibration.py <imagen> --manual  # Manual (clic en 2 puntos)
"""

import sys
import cv2
import numpy as np
from pathlib import Path


def calibrate_automatic(image_path, known_cm=1.0, debug=False):
    """
    Detecta automáticamente marcas de regla y calcula escala.
    
    Args:
        image_path: Ruta a imagen con regla
        known_cm: Distancia entre marcas (default: 1 cm)
        debug: Guardar imagen de debug
        
    Returns:
        Escala en cm/píxel o None si falla
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: No se pudo cargar {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    # Buscar regla en parte inferior (20% de la imagen)
    roi = img[int(h * 0.8):, :]
    
    # Preprocesar
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Detectar líneas verticales
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, 
                            minLineLength=20, maxLineGap=5)
    
    if lines is None:
        print("No se detectaron líneas. Usa --manual")
        return None
    
    # Filtrar líneas verticales
    x_positions = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if 70 < angle < 110:  # Casi vertical
            x_positions.append((x1 + x2) // 2)
    
    if len(x_positions) < 2:
        print(f"Solo {len(x_positions)} marcas detectadas. Usa --manual")
        return None
    
    # Calcular distancias entre marcas consecutivas
    x_positions = sorted(set(x_positions))
    distances = [x_positions[i+1] - x_positions[i] 
                 for i in range(len(x_positions)-1) if x_positions[i+1] - x_positions[i] > 10]
    
    if not distances:
        print("No se pudieron calcular distancias. Usa --manual")
        return None
    
    # Usar mediana
    median_px = np.median(distances)
    scale = known_cm / median_px
    
    print(f"✓ Calibración automática exitosa")
    print(f"  Marcas detectadas: {len(x_positions)}")
    print(f"  Distancia mediana: {median_px:.1f} px")
    print(f"  Escala: {scale:.6f} cm/px")
    
    if debug:
        # Guardar imagen de debug
        vis = roi.copy()
        for x in x_positions:
            cv2.line(vis, (x, 0), (x, roi.shape[0]), (0, 255, 0), 1)
        debug_path = Path(image_path).parent / "calibration_debug.png"
        cv2.imwrite(str(debug_path), vis)
        print(f"  Debug: {debug_path}")
    
    return scale


def calibrate_manual(image_path, known_cm=1.0):
    """
    Calibración manual: el usuario hace clic en 2 puntos.
    
    Args:
        image_path: Ruta a imagen
        known_cm: Distancia entre los puntos (en cm)
        
    Returns:
        Escala en cm/píxel
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"ERROR: No se pudo cargar {image_path}")
        return None
    
    # Redimensionar si es muy grande
    h, w = img.shape[:2]
    max_dim = 1200
    scale_factor = min(max_dim / max(h, w), 1.0)
    if scale_factor < 1:
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    
    points = []
    display = img.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display, str(len(points)), (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(points) == 2:
                cv2.line(display, points[0], points[1], (0, 255, 0), 2)
    
    window = f"Click en 2 puntos separados por {known_cm} cm - ENTER para confirmar"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)
    
    print("\n" + "="*50)
    print("CALIBRACIÓN MANUAL")
    print("="*50)
    print(f"1. Clic en 2 puntos separados {known_cm} cm")
    print("2. ENTER para confirmar")
    print("3. 'r' para reiniciar")
    print("4. ESC para cancelar")
    print("="*50)
    
    while True:
        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            display = img.copy()
            points.clear()
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            return None
        elif key == 13 and len(points) == 2:  # ENTER
            break
    
    cv2.destroyAllWindows()
    
    # Calcular escala
    p1, p2 = points
    distance_px = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) / scale_factor
    scale = known_cm / distance_px
    
    print(f"\n✓ Calibración manual completada")
    print(f"  Distancia: {distance_px:.1f} px")
    print(f"  Escala: {scale:.6f} cm/px")
    
    return scale


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    image_path = sys.argv[1]
    manual = "--manual" in sys.argv
    debug = "--debug" in sys.argv
    
    if manual:
        scale = calibrate_manual(image_path)
    else:
        scale = calibrate_automatic(image_path, debug=debug)
    
    if scale:
        print("\n" + "="*50)
        print("RESULTADO")
        print("="*50)
        print(f"\nActualiza en src/config.py:")
        print(f"    CM_PER_PIXEL = {scale}")


if __name__ == "__main__":
    main()
