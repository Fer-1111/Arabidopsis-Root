# 📋 Guía de Commit a GitHub

## Archivos IMPORTANTES a subir:

### ✅ Archivos principales del proyecto
```
root_analyzer.py          # Script principal ⭐
calibrate.py              # Script de calibración
requirements.txt          # Dependencias
data.yaml                # Config del dataset
README.md                # Documentación principal
.gitignore              # Configuración de Git
```

### ✅ Documentación
```
docs/manual_metodologico.tex    # Manual LaTeX
MEJORAS_PRECISION.md           # Guía de mejoras
```

### ✅ Dataset (imágenes de entrenamiento)
```
train/images/           # Imágenes de entrenamiento
train/labels/           # Labels YOLO
valid/images/           # Validación
valid/labels/
test/images/            # Prueba
test/labels/
```

### ❌ NO subir (ya están en .gitignore):
```
venv/                   # Entorno virtual
*.pt                    # Modelos (muy pesados)
results/                # Salidas generadas
__pycache__/           # Cache de Python
.vscode/               # Configuración del editor
runs/                  # Resultados de entrenamiento
calibration.txt        # Archivo de calibración local
```

## 🚀 Comandos para hacer commit:

### Opción 1: Commit completo (recomendado)
```bash
git add root_analyzer.py calibrate.py requirements.txt data.yaml README.md
git add docs/manual_metodologico.tex MEJORAS_PRECISION.md
git add .gitignore
git commit -m "feat: Sistema completo de análisis de raíces con YOLOv8-seg

- Implementa detección automática con Roboflow API (86.5% mAP)
- Añade esqueletización morfológica para medición precisa
- Ordenamiento espacial izquierda→derecha
- Script de calibración interactiva píxeles→cm
- Salida CSV estructurada y visualización con esqueletos
- Manual metodológico completo en LaTeX
- Procesamiento por lotes de carpetas

Mediciones: 11 raíces detectadas, 0.318 cm promedio
Dataset: 52 imágenes (45 train, 5 valid, 2 test)"
```

### Opción 2: Commits separados por funcionalidad

#### Commit 1: Script principal
```bash
git add root_analyzer.py requirements.txt data.yaml
git commit -m "feat: Script principal de análisis con YOLOv8-seg

- Detección vía Roboflow API
- Esqueletización con scikit-image
- Ordenamiento espacial
- Salida CSV y visualización
- Procesamiento por lotes"
```

#### Commit 2: Calibración
```bash
git add calibrate.py
git commit -m "feat: Herramienta de calibración interactiva

Permite calcular CM_PER_PIXEL con interfaz gráfica
Marca 2 puntos y calcula escala automáticamente"
```

#### Commit 3: Documentación
```bash
git add docs/manual_metodologico.tex MEJORAS_PRECISION.md README.md
git commit -m "docs: Manual metodológico completo

- Manual LaTeX con fundamento teórico
- Guía de mejoras de precisión
- README actualizado con ejemplos de uso
- Referencias a RootNav 2.0"
```

#### Commit 4: Configuración
```bash
git add .gitignore
git commit -m "chore: Actualizar .gitignore para resultados"
```

### Push a GitHub
```bash
git push origin main
```

## 📝 Convenciones de commits (si quieres seguir estándares):

- `feat:` - Nueva funcionalidad
- `fix:` - Corrección de bug
- `docs:` - Cambios en documentación
- `style:` - Formato de código (sin cambios funcionales)
- `refactor:` - Refactorización de código
- `test:` - Añadir/modificar tests
- `chore:` - Mantenimiento (dependencias, config)

## 🔍 Verificar antes de commit:

```bash
# Ver estado actual
git status

# Ver diferencias
git diff

# Ver archivos que se subirán
git diff --cached

# Ver tamaño de archivos
du -sh venv/ results/ *.pt
```

## ⚠️ Importante:

1. **NO subir modelos .pt** (muy pesados, usar .gitignore)
2. **NO subir venv/** (cada usuario crea su propio entorno)
3. **NO subir results/** (son salidas generadas, no código)
4. **SÍ subir el dataset** (train/valid/test con imágenes y labels)
5. **SÍ subir requirements.txt** (para que otros puedan instalar)

## 🎯 Mensaje de commit sugerido (copy-paste):

```
feat: Sistema completo de análisis de raíces Arabidopsis

Implementa pipeline automatizado de detección y medición:
- YOLOv8-seg con Roboflow API (mAP 86.5%)
- Esqueletización morfológica (Zhang-Suen)
- Calibración interactiva píxeles→cm
- Ordenamiento espacial izq→der
- Salida CSV + visualización magenta
- Procesamiento por lotes
- Manual metodológico LaTeX

Performance: 21 raíces en 2 imágenes, 0.309cm promedio
Dataset: 52 imágenes anotadas (Roboflow format)
```

## 🔗 Después del push:

1. Ve a GitHub.com y verifica que se subió correctamente
2. Añade una descripción al repositorio
3. Opcionalmente añade topics: `yolov8`, `computer-vision`, `plant-phenotyping`, `arabidopsis`
4. Considera añadir una licencia (MIT recomendada)
5. Añade una imagen demo en el README
