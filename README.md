# 🌱 Arabidopsis Root Detection

Sistema automatizado para **detección y medición de raíces primarias** en plántulas de *Arabidopsis thaliana* usando YOLOv8-seg y visión por computadora.

## 📋 Descripción

Este proyecto automatiza la medición del crecimiento radicular en estudios de fenotipado vegetal mediante:

- **YOLOv8-seg** para segmentación semántica de raíces
- **Esqueletización morfológica** para medición precisa de longitud
- **Calibración automática/manual** de escala espacial

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/Fer-1111/ArabidopsisRootDetection.git
cd ArabidopsisRootDetection

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 📁 Estructura

```
ArabidopsisRootDetection/
├── src/
│   ├── config.py              # Configuración
│   ├── ruler_calibration.py   # Calibración de escala
│   └── train_and_measure.py   # Pipeline principal
├── data/
│   ├── train/                 # Imágenes de entrenamiento
│   ├── valid/                 # Imágenes de validación
│   ├── test/                  # Imágenes de test
│   └── data.yaml              # Configuración del dataset
├── models/                    # Modelos entrenados (.pt)
├── results/                   # Mediciones y resultados
└── requirements.txt
```

## 🔧 Uso

### 1. Entrenar modelo

```bash
python src/train_and_measure.py train

# Con número específico de épocas
python src/train_and_measure.py train 100
```

### 2. Evaluar modelo

```bash
python src/train_and_measure.py evaluate
```

### 3. Medir raíces

```bash
# Una imagen
python src/train_and_measure.py measure imagen.jpg

# Múltiples imágenes
python src/train_and_measure.py batch carpeta/
```

### 4. Calibrar escala

```bash
# Automático (detecta regla)
python src/ruler_calibration.py imagen_con_regla.jpg

# Manual (clic en 2 puntos)
python src/ruler_calibration.py imagen.jpg --manual
```

## 📊 Pipeline

```
Imagen → YOLOv8-seg → Máscara → Esqueletización → Longitud (cm)
                                      ↑
                              Calibración (cm/px)
```

## 📈 Métricas

| Métrica | Descripción |
|---------|-------------|
| mAP50 | Mean Average Precision @ IoU 50% |
| mAP50-95 | mAP promediado IoU 50-95% |

## 🗃️ Dataset

- **Fuente**: Roboflow
- **Clase**: `root` (raíz primaria)
- **Formato**: YOLOv8 Segmentation

## 📝 Configuración

Edita `src/config.py` para ajustar:

```python
CM_PER_PIXEL = 0.005          # Escala de calibración
TRAINING_CONFIG = {
    "epochs": 75,
    "imgsz": 640,
    "batch": 4,
}
```

## 📄 Licencia

MIT License
