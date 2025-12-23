"""
root_analyzer.py - Análisis simple de raíces con YOLOv8-seg
===========================================================

Detecta raíces, las esqueletiza, mide y ordena de izquierda a derecha.

Uso:
    python root_analyzer.py <imagen>
    python root_analyzer.py <carpeta>
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import label
from ultralytics import YOLO

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# IMPORTANTE: Cambiar a tu modelo entrenado
# Opción 1: Modelo local (si tienes el .pt descargado)
# MODEL_PATH = "models/best.pt"
# USE_ROBOFLOW_API = False

# Opción 2: API de Roboflow (modelo ya entrenado en la nube)
MODEL_PATH = None
USE_ROBOFLOW_API = True
ROBOFLOW_API_KEY = "RXnQJhwNYU896IpBkehH"
ROBOFLOW_MODEL_ID = "arabidopsis-primary-root-2bwll/2"

# Parámetros de detección (ajusta para mejorar precisión)
CM_PER_PIXEL = 0.004           # Calibración (ajusta con tu regla)
CONFIDENCE = 0.5               # Umbral de confianza mínima (0.3-0.7 recomendado)
IOU_THRESHOLD = 0.3            # Umbral IoU para NMS (0.3-0.5 recomendado)
MIN_ROOT_LENGTH_CM = 0.1       # Filtro de ruido: ignorar raíces < 0.1 cm

# Visualización
SKELETON_COLOR = (255, 0, 255)  # Magenta (BGR)
SKELETON_THICKNESS = 2

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# FUNCIONES
# ============================================================================

def measure_root(mask, scale=CM_PER_PIXEL):
    """
    Mide una raíz usando esqueletización.
    
    Returns:
        length_cm: Longitud en centímetros
        length_px: Longitud en píxeles
        skeleton: Imagen del esqueleto (para visualización)
    """
    if mask is None or np.sum(mask) == 0:
        return 0.0, 0, None
    
    # Asegurar que sea binaria
    binary_mask = (mask > 0).astype(bool)
    
    # Esqueletizar
    skeleton = skeletonize(binary_mask)
    
    # Contar píxeles del esqueleto
    length_px = int(np.sum(skeleton))
    
    # Convertir a cm
    length_cm = length_px * scale
    
    # Convertir skeleton a uint8 para visualización
    skeleton_img = (skeleton * 255).astype(np.uint8)
    
    return length_cm, length_px, skeleton_img


def get_root_position_x(mask):
    """
    Obtiene la posición X del centro de la raíz.
    Usado para ordenar de izquierda a derecha.
    """
    # Encontrar el centroide
    M = cv2.moments(mask)
    if M["m00"] == 0:
        return 0
    cx = int(M["m10"] / M["m00"])
    return cx


def process_image(model, image_path, use_roboflow=False):
    """
    Procesa una imagen: detecta raíces, esqueletiza, mide.
    
    Returns:
        measurements: Lista de diccionarios con mediciones
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"❌ No encontrado: {image_path}")
        return []
    
    print(f"\n📷 Procesando: {image_path.name}")
    
    # Leer imagen
    img = cv2.imread(str(image_path))
    if img is None:
        print("   ❌ Error al leer imagen")
        return []
    
    h, w = img.shape[:2]
    
    # Inferencia
    try:
        if use_roboflow:
            # Usar API de Roboflow
            from inference_sdk import InferenceHTTPClient
            result_data = model.infer(str(image_path), model_id=ROBOFLOW_MODEL_ID)
            
            # Convertir respuesta de Roboflow a formato compatible
            if "predictions" not in result_data or len(result_data["predictions"]) == 0:
                print("   ⚠️  No se detectaron raíces")
                return []
            
            masks_list = []
            confidences = []
            
            # Filtrar por confianza manualmente
            for pred in result_data["predictions"]:
                conf = pred.get("confidence", 0)
                if conf < CONFIDENCE:  # Aplicar umbral de confianza
                    continue
                    
                if "points" in pred and pred["points"]:
                    # Crear máscara desde polígono
                    pts = np.array([[p["x"], p["y"]] for p in pred["points"]], dtype=np.int32)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 255)
                    masks_list.append(mask)
                    confidences.append(conf)
        else:
            # Usar modelo local YOLO
            results = model(str(image_path), conf=CONFIDENCE, iou=IOU_THRESHOLD, verbose=False)
            result = results[0]
            
            if result.masks is None or len(result.masks.data) == 0:
                print("   ⚠️  No se detectaron raíces")
                return []
            
            masks_list = []
            confidences = []
            for mask_data, conf in zip(result.masks.data, result.boxes.conf):
                mask = mask_data.cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) * 255
                masks_list.append(mask)
                confidences.append(float(conf))
    
    except Exception as e:
        print(f"   ❌ Error en inferencia: {e}")
        return []
    
    print(f"   ✓ Detectadas {len(masks_list)} raíces")
    
    # Procesar cada raíz
    roots_data = []
    
    for i, (mask, conf) in enumerate(zip(masks_list, confidences)):
        # Medir longitud (con esqueletización)
        length_cm, length_px, skeleton = measure_root(mask)
        
        # Filtrar ruido: ignorar detecciones muy pequeñas
        if length_cm < MIN_ROOT_LENGTH_CM:
            continue
        
        # Obtener posición X para ordenar
        pos_x = get_root_position_x(mask)
        
        roots_data.append({
            "mask": mask,
            "skeleton": skeleton,
            "pos_x": pos_x,
            "length_cm": length_cm,
            "length_px": length_px,
            "confidence": conf
        })
    
    # Ordenar de izquierda a derecha por posición X
    roots_data.sort(key=lambda x: x["pos_x"])
    
    # Crear mediciones finales
    measurements = []
    for idx, root in enumerate(roots_data, start=1):
        measurements.append({
            "image": image_path.name,
            "root_id": idx,  # 1, 2, 3... de izquierda a derecha
            "position_x": root["pos_x"],
            "length_cm": round(root["length_cm"], 3),
            "length_px": root["length_px"],
            "confidence": round(root["confidence"], 3),
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"   Raíz {idx} (x={root['pos_x']:4d}): {root['length_cm']:.3f} cm ({root['length_px']} px)")
    
    # Guardar visualización
    save_visualization(img, roots_data, image_path.name)
    
    return measurements


def save_visualization(original_img, roots_data, filename):
    """
    Guarda imagen con esqueletos superpuestos en magenta.
    """
    h, w = original_img.shape[:2]
    
    # Crear imagen de salida
    output = original_img.copy()
    
    # Crear capa de esqueletos combinados
    skeleton_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Superponer todos los esqueletos en magenta
    for idx, root in enumerate(roots_data):
        skeleton = root["skeleton"]
        if skeleton is not None:
            # Dibujar esqueleto en magenta
            skeleton_overlay[skeleton > 0] = SKELETON_COLOR
    
    # Hacer el esqueleto más visible con dilatación
    kernel = np.ones((SKELETON_THICKNESS, SKELETON_THICKNESS), np.uint8)
    skeleton_overlay = cv2.dilate(skeleton_overlay, kernel, iterations=1)
    
    # Superponer con transparencia
    alpha = 0.8
    output = cv2.addWeighted(output, 1, skeleton_overlay, alpha, 0)
    
    # Agregar números de raíz
    for idx, root in enumerate(roots_data):
        # Posición para el texto (arriba del esqueleto)
        x_pos = root["pos_x"]
        cv2.putText(output, f"#{idx+1}", (x_pos, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, SKELETON_COLOR, 2)
    
    # Guardar
    output_path = RESULTS_DIR / f"skeleton_{filename}"
    cv2.imwrite(str(output_path), output)
    print(f"   💾 Guardado: {output_path}")


def save_csv(all_measurements):
    """Guarda mediciones en CSV con formato mejorado."""
    if not all_measurements:
        print("\n⚠️  No hay mediciones para guardar")
        return
    
    csv_path = RESULTS_DIR / "root_measurements.csv"
    
    # Reorganizar columnas para mejor legibilidad
    fieldnames = [
        "timestamp",
        "image", 
        "plant_id",      # Cambiado de root_id para claridad
        "position_x",
        "length_cm",
        "length_px",
        "confidence"
    ]
    
    # Renombrar root_id a plant_id
    for m in all_measurements:
        m["plant_id"] = m.pop("root_id")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_measurements)
    
    # Estadísticas
    total_roots = len(all_measurements)
    avg_length = sum(m["length_cm"] for m in all_measurements) / total_roots
    min_length = min(m["length_cm"] for m in all_measurements)
    max_length = max(m["length_cm"] for m in all_measurements)
    
    print(f"\n✅ CSV guardado: {csv_path}")
    print(f"\n📊 Estadísticas:")
    print(f"   Total raíces: {total_roots}")
    print(f"   Longitud promedio: {avg_length:.3f} cm")
    print(f"   Rango: {min_length:.3f} - {max_length:.3f} cm")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nEjemplos:")
        print("  python root_analyzer.py imagen.jpg")
        print("  python root_analyzer.py test/images/")
        print(f"\nConfiguración actual:")
        print(f"  • Modelo: {'Roboflow API' if USE_ROBOFLOW_API else MODEL_PATH}")
        print(f"  • Escala: {CM_PER_PIXEL} cm/píxel")
        print(f"  • Confianza mínima: {CONFIDENCE}")
        return
    
    print("=" * 70)
    print("  ANÁLISIS DE RAÍCES - YOLOv8-seg + Esqueletización")
    print("=" * 70)
    print(f"Parámetros: Conf={CONFIDENCE} | IoU={IOU_THRESHOLD} | Min={MIN_ROOT_LENGTH_CM}cm")
    print()
    
    # Cargar modelo
    if USE_ROBOFLOW_API:
        print(f"🌐 Usando Roboflow API: {ROBOFLOW_MODEL_ID}...")
        try:
            from inference_sdk import InferenceHTTPClient
            model = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=ROBOFLOW_API_KEY
            )
            print("✓ Cliente API inicializado\n")
        except Exception as e:
            print(f"❌ Error al inicializar API: {e}")
            return
    else:
        print(f"Cargando modelo: {MODEL_PATH}...")
        try:
            model = YOLO(MODEL_PATH)
            print("✓ Modelo cargado\n")
        except Exception as e:
            print(f"❌ Error al cargar modelo: {e}")
            return
    
    # Procesar
    path = Path(sys.argv[1])
    all_measurements = []
    
    if path.is_file():
        # Una imagen
        all_measurements.extend(process_image(model, str(path), use_roboflow=USE_ROBOFLOW_API))
    
    elif path.is_dir():
        # Carpeta completa
        images = sorted(list(path.glob("*.jpg")) + list(path.glob("*.png")))
        print(f"📁 Encontradas {len(images)} imágenes")
        
        for img_path in images:
            all_measurements.extend(process_image(model, str(img_path), use_roboflow=USE_ROBOFLOW_API))
    
    else:
        print(f"❌ No es archivo ni carpeta: {path}")
        return
    
    # Guardar resultados
    save_csv(all_measurements)
    
    print("\n" + "=" * 70)
    print("  ✅ ANÁLISIS COMPLETADO")
    print("=" * 70)
    print(f"\nResultados en: {RESULTS_DIR}/")
    print("  • root_measurements.csv  ← Mediciones ordenadas")
    print("  • skeleton_*.jpg         ← Visualización de esqueletos")


if __name__ == "__main__":
    main()
