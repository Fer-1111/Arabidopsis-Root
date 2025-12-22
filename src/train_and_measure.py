"""
train_and_measure.py - Pipeline principal para detección de raíces
==================================================================

Uso:
    python src/train_and_measure.py train          # Entrenar modelo
    python src/train_and_measure.py evaluate       # Evaluar en test set
    python src/train_and_measure.py measure IMG    # Medir raíces en imagen
    python src/train_and_measure.py batch FOLDER   # Medir múltiples imágenes
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO
from skimage.morphology import skeletonize

# Agregar src al path para imports
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_YAML, MODELS_DIR, RESULTS_DIR,
    TRAINING_CONFIG, INFERENCE_CONFIG, CM_PER_PIXEL
)


class RootDetector:
    """Detector y medidor de raíces usando YOLOv8-seg."""
    
    def __init__(self, model_path=None, scale=CM_PER_PIXEL):
        """
        Args:
            model_path: Ruta al modelo .pt (None = buscar en models/)
            scale: Escala cm/píxel para mediciones
        """
        self.scale = scale
        self.model = None
        
        # Buscar modelo
        if model_path and Path(model_path).exists():
            self.model_path = Path(model_path)
        else:
            # Buscar en carpeta models/
            self.model_path = self._find_model()
        
        if self.model_path:
            self.model = YOLO(str(self.model_path))
            print(f"✓ Modelo cargado: {self.model_path}")
        else:
            print("⚠ No hay modelo entrenado. Ejecuta 'train' primero.")
    
    def _find_model(self):
        """Busca el mejor modelo en models/"""
        MODELS_DIR.mkdir(exist_ok=True)
        
        # Buscar best.pt
        best = MODELS_DIR / "best.pt"
        if best.exists():
            return best
        
        # Buscar en runs/
        runs_best = Path("runs/segment/train/weights/best.pt")
        if runs_best.exists():
            return runs_best
        
        return None
    
    def train(self, epochs=None, **kwargs):
        """Entrena el modelo."""
        if not DATA_YAML.exists():
            print(f"ERROR: No se encontró {DATA_YAML}")
            print("Asegúrate de tener el dataset en data/")
            return None
        
        config = {**TRAINING_CONFIG, **kwargs}
        if epochs:
            config["epochs"] = epochs
        
        print("="*60)
        print("ENTRENAMIENTO YOLOv8-seg")
        print("="*60)
        print(f"Dataset: {DATA_YAML}")
        print(f"Épocas: {config['epochs']}")
        print("="*60)
        
        # Cargar modelo base
        model = YOLO(config["model"])
        
        # Entrenar
        results = model.train(
            data=str(DATA_YAML),
            epochs=config["epochs"],
            imgsz=config["imgsz"],
            batch=config["batch"],
            patience=config["patience"],
            device=config.get("device", ""),
            project="runs/segment",
            name="train",
            exist_ok=True
        )
        
        # Copiar mejor modelo a models/
        trained_model = Path("runs/segment/train/weights/best.pt")
        if trained_model.exists():
            MODELS_DIR.mkdir(exist_ok=True)
            import shutil
            shutil.copy(trained_model, MODELS_DIR / "best.pt")
            print(f"\n✓ Modelo guardado en: {MODELS_DIR / 'best.pt'}")
        
        return results
    
    def evaluate(self):
        """Evalúa el modelo en test set."""
        if self.model is None:
            print("ERROR: No hay modelo cargado")
            return
        
        print("="*60)
        print("EVALUACIÓN")
        print("="*60)
        
        metrics = self.model.val(data=str(DATA_YAML), split="test")
        
        print(f"\nmAP50 (segmentación): {metrics.seg.map50:.4f}")
        print(f"mAP50-95 (segmentación): {metrics.seg.map:.4f}")
        
        return metrics
    
    def measure_image(self, image_path, save_output=True):
        """
        Mide raíces en una imagen.
        
        Returns:
            Lista de diccionarios con mediciones
        """
        if self.model is None:
            print("ERROR: No hay modelo cargado")
            return []
        
        image_path = Path(image_path)
        if not image_path.exists():
            print(f"ERROR: No existe {image_path}")
            return []
        
        print(f"\nProcesando: {image_path.name}")
        
        # Inferencia
        results = self.model(
            str(image_path),
            conf=INFERENCE_CONFIG["confidence"],
            iou=INFERENCE_CONFIG["iou"],
            verbose=False
        )
        
        result = results[0]
        
        if result.masks is None or len(result.masks.data) == 0:
            print("  No se detectaron raíces")
            return []
        
        # Cargar imagen
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        measurements = []
        
        for i, (mask_data, conf) in enumerate(zip(result.masks.data, result.boxes.conf)):
            # Procesar máscara
            mask = mask_data.cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) * 255
            
            # Esqueletizar y medir
            skeleton = skeletonize(mask > 0)
            length_px = int(np.sum(skeleton))
            length_cm = length_px * self.scale
            
            measurement = {
                "image": image_path.name,
                "root_id": i + 1,
                "length_cm": round(length_cm, 3),
                "length_px": length_px,
                "confidence": round(float(conf), 3),
                "timestamp": datetime.now().isoformat()
            }
            measurements.append(measurement)
            
            print(f"  Raíz {i+1}: {length_cm:.2f} cm ({length_px} px)")
        
        # Guardar imagen anotada
        if save_output:
            RESULTS_DIR.mkdir(exist_ok=True)
            annotated = result.plot()
            
            # Agregar texto con mediciones
            y = 30
            for m in measurements:
                text = f"Raiz {m['root_id']}: {m['length_cm']:.2f} cm"
                cv2.putText(annotated, text, (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 25
            
            output_path = RESULTS_DIR / f"measured_{image_path.name}"
            cv2.imwrite(str(output_path), annotated)
            print(f"  Guardado: {output_path}")
        
        return measurements
    
    def measure_batch(self, folder_path, pattern="*.jpg"):
        """Mide múltiples imágenes y guarda CSV."""
        folder = Path(folder_path)
        images = list(folder.glob(pattern)) + list(folder.glob("*.png"))
        
        if not images:
            print(f"No se encontraron imágenes en {folder}")
            return []
        
        print(f"Procesando {len(images)} imágenes...")
        
        all_measurements = []
        for img_path in images:
            measurements = self.measure_image(img_path)
            all_measurements.extend(measurements)
        
        # Guardar CSV
        if all_measurements:
            RESULTS_DIR.mkdir(exist_ok=True)
            csv_path = RESULTS_DIR / "root_measurements.csv"
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_measurements[0].keys())
                writer.writeheader()
                writer.writerows(all_measurements)
            
            print(f"\n✓ Resultados guardados en: {csv_path}")
        
        return all_measurements


def main():
    """CLI principal."""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    detector = RootDetector()
    
    if command == "train":
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else None
        detector.train(epochs=epochs)
    
    elif command == "evaluate":
        detector.evaluate()
    
    elif command == "measure":
        if len(sys.argv) < 3:
            print("Uso: python train_and_measure.py measure <imagen>")
            return
        detector.measure_image(sys.argv[2])
    
    elif command == "batch":
        if len(sys.argv) < 3:
            print("Uso: python train_and_measure.py batch <carpeta>")
            return
        detector.measure_batch(sys.argv[2])
    
    else:
        print(f"Comando desconocido: {command}")
        print("Comandos: train, evaluate, measure, batch")


if __name__ == "__main__":
    main()
