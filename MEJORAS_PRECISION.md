# 🎯 Guía de Mejora de Precisión - Inspirada en RootNav 2.0

## 📊 Estado Actual
- ✅ **Modelo funcionando**: 86.5% mAP en Roboflow
- ✅ **Esqueletización**: Implementada con scikit-image
- ✅ **Ordenamiento**: Por posición X (izquierda a derecha)
- ✅ **Visualización**: Esqueletos en magenta con numeración

---

## 🔧 Ajustes Inmediatos para Mejorar Precisión

### 1. **Ajustar Hiperparámetros de Detección**

Edita las siguientes líneas en `root_analyzer.py`:

```python
# Línea 42-44: Ajustar confianza e IoU
CONFIDENCE = 0.4        # Bajar de 0.5 → detecta más raíces (puede incluir falsos positivos)
                        # Subir a 0.6-0.7 → más estricto (puede perder raíces pequeñas)

IOU_THRESHOLD = 0.4     # Umbral de superposición para eliminar duplicados
                        # Más bajo (0.2) = menos duplicados, más agresivo
                        # Más alto (0.5) = permite más superposición

MIN_LENGTH_CM = 0.05    # Filtrar raíces menores a 0.5 mm (elimina ruido)
```

**Recomendación para tus datos:**
- Si **falta detectar raíces**: `CONFIDENCE = 0.3`, `MIN_LENGTH_CM = 0.05`
- Si **detecta ruido/duplicados**: `CONFIDENCE = 0.6`, `IOU_THRESHOLD = 0.5`

---

### 2. **Mejorar Calibración de Escala**

Actualmente: `CM_PER_PIXEL = 0.004` (¿manual?)

**Método automático con regla** (del script `eskeleto.py`):

```python
# Usa una imagen con regla de referencia conocida
KNOWN_LENGTH_CM = 1.0  # 1 cm en la regla
MEASURED_PIXELS = 250  # Píxeles que ocupa ese 1 cm en tu imagen

CM_PER_PIXEL = KNOWN_LENGTH_CM / MEASURED_PIXELS
# Resultado: 0.004 cm/pixel
```

**Verificación**: Mide una raíz manualmente en ImageJ/Fiji y compara con el resultado del script.

---

### 3. **Post-procesamiento de Esqueletos** (Inspirado en RootNav 2.0)

Actualmente el esqueleto puede tener bifurcaciones. RootNav 2.0 recomienda:

#### a) **Pruning (Poda de ramas cortas)**

Agrega después de la línea 68 en `root_analyzer.py`:

```python
def measure_root(mask, scale_cm_per_pixel=CM_PER_PIXEL):
    """Mide longitud usando esqueletización."""
    if mask is None or np.sum(mask) == 0:
        return 0.0, 0, None
    
    mask_binary = (mask > 0).astype(np.uint8)
    skeleton = skeletonize(mask_binary)
    
    # ====== NUEVA SECCIÓN: Poda de ramas laterales ======
    from scipy.ndimage import label
    
    # Etiquetar componentes conectados
    labeled, num_features = label(skeleton)
    
    # Si hay múltiples componentes, quedarse con el más largo
    if num_features > 1:
        component_lengths = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        longest_component = np.argmax(component_lengths) + 1
        skeleton = (labeled == longest_component).astype(np.uint8)
    # ===================================================
    
    length_px = np.sum(skeleton)
    length_cm = length_px * scale_cm_per_pixel
    
    return length_cm, length_px, skeleton
```

#### b) **Suavizado morfológico**

Antes de la línea 66:

```python
# Cerrar huecos pequeños en la máscara
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
```

---

## 🚀 Mejoras Avanzadas (Requieren Reentrenamiento)

### 4. **Aumentar el Dataset** (RootNav 2.0 usó 42,000 imágenes)

Tu dataset actual: **52 imágenes** (pequeño para deep learning)

**Opciones:**

#### a) **Data Augmentation en Roboflow**
Ya aplicado según `data.yaml`:
- ✅ Flip horizontal/vertical
- ✅ Rotación ±15°
- ✅ Brillo ±25%
- ✅ Blur 0-2px

**Agregar más**:
- Shear (cizallamiento)
- Zoom ±10%
- Ruido gaussiano
- Cambios de saturación

#### b) **Obtener más imágenes reales**
- Fotografiar más plántulas (objetivo: >200 imágenes únicas)
- Incluir variedad de condiciones:
  - Diferentes edades (3, 5, 7 días)
  - Diferentes tratamientos (control, sorbitol, etc.)
  - Diferentes iluminaciones
  - Diferentes densidades de siembra

#### c) **Imágenes sintéticas** (como RootNav 2.0)
- Generar raíces artificiales con curvas Bézier
- Aplicar texturas realistas
- Herramientas: `imgaug`, `albumentations`

---

### 5. **Re-entrenar con Más Épocas**

Tu modelo actual: entrenamiento en Roboflow (~26 min)

**Para entrenar localmente** (mejor control):

```bash
# Crear script de entrenamiento mejorado
python train_locally.py
```

```python
# train_locally.py
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # Partir del modelo base

results = model.train(
    data='data.yaml',
    epochs=200,              # Más épocas (tu modelo: ~75)
    imgsz=640,
    batch=4,                 # Ajustar según tu GPU (4, 8, 16)
    patience=30,             # Early stopping
    optimizer='AdamW',       # Mejor que SGD para datasets pequeños
    lr0=0.001,               # Learning rate inicial
    augment=True,            # Data augmentation adicional
    mosaic=1.0,              # Mosaic augmentation (efecto RootNav)
    mixup=0.1,               # Mezclar imágenes
    copy_paste=0.1,          # Copy-paste augmentation
    degrees=20,              # Rotación aumentada
    flipud=0.5,              # Flip vertical
    fliplr=0.5,              # Flip horizontal
    hsv_h=0.015,             # Variación de tono
    hsv_s=0.7,               # Saturación
    hsv_v=0.4,               # Valor/Brillo
)
```

**Ventajas:**
- Control total sobre hiperparámetros
- Monitoreo de métricas en tiempo real
- Posibilidad de fine-tuning

---

### 6. **Usar Modelo Más Grande**

Actualmente: `yolov8n-seg.pt` (nano, 3.4 MB, más rápido)

**Alternativas:**

| Modelo | Tamaño | Velocidad | Precisión | Uso recomendado |
|--------|--------|-----------|-----------|-----------------|
| yolov8n-seg | 3.4 MB | ⚡⚡⚡ | ⭐⭐ | Prototipado rápido |
| yolov8s-seg | 11.8 MB | ⚡⚡ | ⭐⭐⭐ | **Ideal para tus datos** |
| yolov8m-seg | 27.3 MB | ⚡ | ⭐⭐⭐⭐ | Dataset > 200 imágenes |
| yolov8l-seg | 46.0 MB | 🐌 | ⭐⭐⭐⭐⭐ | Máxima precisión |

**Recomendación:** Probar `yolov8s-seg` (small) para mejor equilibrio.

```bash
# Cambiar en el entrenamiento
model = YOLO('yolov8s-seg.pt')
```

---

### 7. **Implementar Tracking entre Imágenes** (Inspirado en tu `eskeleto.py`)

Para experimentos longitudinales (mismas plantas en días diferentes):

```python
def assign_persistent_ids(current_positions, previous_positions, threshold=50):
    """
    Asigna IDs persistentes basándose en posiciones X previas.
    
    Args:
        current_positions: Lista de posiciones X actuales
        previous_positions: Diccionario {plant_id: x_position} del día anterior
        threshold: Distancia máxima en píxeles para considerar la misma planta
    
    Returns:
        Diccionario {current_index: plant_id}
    """
    assignments = {}
    used_ids = set()
    
    for i, curr_x in enumerate(current_positions):
        best_match = None
        min_distance = threshold
        
        for prev_id, prev_x in previous_positions.items():
            if prev_id in used_ids:
                continue
            distance = abs(curr_x - prev_x)
            if distance < min_distance:
                min_distance = distance
                best_match = prev_id
        
        if best_match is not None:
            assignments[i] = best_match
            used_ids.add(best_match)
        else:
            # Nueva planta (no estaba en imagen previa)
            new_id = max(previous_positions.keys(), default=0) + 1
            assignments[i] = new_id
    
    return assignments

# Uso en el CSV:
# Día 1: Control_1.jpg → plant_id=1, plant_id=2, ...
# Día 3: Control_3.jpg → Reasignar IDs basándose en posiciones previas
```

---

## 📈 Estrategia de Mejora Recomendada

### Fase 1: **Ajustes Inmediatos** (1-2 horas)
1. ✅ Ajustar `CONFIDENCE = 0.4`
2. ✅ Verificar calibración con regla física
3. ✅ Implementar pruning de esqueletos
4. ✅ Probar en 10 imágenes de validación

### Fase 2: **Expansión de Dataset** (1 semana)
1. Fotografiar 100 nuevas imágenes
2. Anotar en Roboflow (usar SAM auto-annotation)
3. Re-entrenar con dataset expandido

### Fase 3: **Modelo Avanzado** (1-2 semanas)
1. Entrenar `yolov8s-seg` localmente con 200 épocas
2. Implementar tracking longitudinal
3. Validar con experimento completo

---

## 📚 Referencias Clave de RootNav 2.0

De tu paper mencionado:

1. **Arquitectura**: U-Net con ResNet-50 (vs. YOLOv8-seg)
   - YOLOv8 es más eficiente y comparable en precisión
   
2. **Dataset**: 42,000 imágenes sintéticas + reales
   - Tú: 52 reales → **Prioridad: aumentar dataset**
   
3. **Post-procesamiento**: Thinning (esqueletización) + navegación de grafos
   - ✅ Ya implementado con `skeletonize()`
   
4. **Métricas**: Precision/Recall en píxeles
   - Roboflow te da mAP (mean Average Precision) = 86.5% ✅
   
5. **Tracking**: Comparación frame-a-frame
   - Implementar para estudios longitudinales

---

## 🛠️ Herramientas Adicionales

- **Validación manual**: [LabelImg](https://github.com/heartexlabs/labelImg) para revisar anotaciones
- **Análisis morfológico**: [ImageJ/Fiji](https://imagej.net/software/fiji/) para verificar mediciones
- **Augmentation**: [Albumentations](https://albumentations.ai/) para generar variaciones
- **Visualización**: [Weights & Biases](https://wandb.ai/) para monitorear entrenamientos

---

## ✅ Checklist de Verificación

Antes de procesar tu dataset completo:

- [ ] Verificar calibración con regla (medir objeto conocido)
- [ ] Probar diferentes valores de `CONFIDENCE` (0.3, 0.4, 0.5, 0.6)
- [ ] Validar visualmente 20 imágenes procesadas
- [ ] Comparar longitudes con mediciones manuales (error < 5%)
- [ ] Documentar parámetros óptimos encontrados

---

**¿Cuál de estas mejoras te gustaría implementar primero?** 🚀
