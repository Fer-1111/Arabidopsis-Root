"""
config.py - Configuración central del proyecto
"""

from pathlib import Path

# =============================================================================
# RUTAS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Archivo de configuración del dataset
DATA_YAML = DATA_DIR / "data.yaml"

# =============================================================================
# ENTRENAMIENTO
# =============================================================================

TRAINING_CONFIG = {
    "model": "yolov8n-seg.pt",   # Modelo base
    "epochs": 75,
    "imgsz": 640,
    "batch": 4,
    "patience": 15,
    "device": "",                 # "" = auto, "cpu", "0" = GPU 0
}

# =============================================================================
# INFERENCIA
# =============================================================================

INFERENCE_CONFIG = {
    "confidence": 0.7,
    "iou": 0.5,
}

# =============================================================================
# CALIBRACIÓN
# =============================================================================

# Escala por defecto (cm por píxel)
# Ajustar con ruler_calibration.py para tu setup específico
CM_PER_PIXEL = 0.005
