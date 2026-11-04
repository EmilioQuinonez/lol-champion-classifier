# LoL Champions Visual Data

Red neuronal convolucional hecha con TensorFlow / Keras que reconoce a cuál de 38 campeones de League of Legends pertenece una *splash art* y regresa su top 5 de campeones más probables. Se sigue un proceso incremental de ocho versiones: una CNN desde cero, aumento de datos con Dropout, transfer learning con EfficientNetB0 (congelada y con fine-tuning), una entrada de tres recortes a 384 px, entrenamiento en escala de grises y, para la interfaz web, reentrenamiento con todos los datos.

## Archivos

- `notebooks/data_setup.ipynb`: limpieza del dataset (solo splash arts, detección de splash grupales y duplicados (MD5 y average hash), filtro de campeones con al menos 15 imágenes y split train/val/test estratificado por campeón).
- `notebooks/training.ipynb`: entrenamiento y evaluación de las versiones v1–v8, análisis del modelo final en test y guardado del modelo de la interfaz web.
- `src/app.py` y `src/web/`: interfaz web para subir imágenes y ver el top 5 del modelo.
- `docs/reporte.tex` / `.pdf`: reporte del proyecto, con las gráficas en `docs/figures/`.
- `experimentos/`: resultados de las iteraciones anteriores al notebook de entrenamiento (107 campeones, comparación de backbones y formas de pasar a cuadrado, multi-recorte y escala de grises), usados en el reporte.
- `test_images/`: imágenes externas para probar la interfaz web.
- `models/`: modelo de la interfaz web (`lol_champion_classifier.keras`, la v8) y su `metadata.json`. Las versiones v1–v8 (`models/versions/`) no se suben por su tamaño.
- `data/`: dataset ([fuente en Kaggle](https://www.kaggle.com/datasets/houssemeddineelimam/league-of-legends-champions-visual-data)). No se sube al repo por su tamaño.

## Cómo correrlo

Requiere Python 3.11. Primero se crea un entorno virtual y se instalan las dependencias:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

El modelo ya viene en el repo, así que para probar la interfaz web basta con:

```bash
python src/app.py   # abre http://localhost:8000
```

Para reproducir el entrenamiento hay que seleccionar el kernel de `.venv` en Jupyter o VS Code y seguir estos pasos:

1. Descarga el dataset de Kaggle y descomprímelo en `data/`, de modo que quede `data/lol_champions_visual_data/<campeón>/`.
2. Corre `notebooks/data_setup.ipynb`, que genera `data/processed/` con los splits.
3. Corre `notebooks/training.ipynb`, que guarda cada versión en `models/versions/` y el modelo de la interfaz web en `models/lol_champion_classifier.keras` junto con `models/metadata.json`.

## Resultado

Con 38 campeones y ~12 splash arts por campeón en train, una CNN desde cero memoriza train (100%) y se queda en 0.105 de accuracy en validación. El aumento de datos con Dropout no la mejora, y el transfer learning con EfficientNetB0 llega a 0.200 con fine-tuning. Todas las decisiones se tomaron con la accuracy de validación, sin tocar test.

El cambio que más ayudó no fue de arquitectura sino de entrada: dividir la splash en tres recortes cuadrados de 384 px, en lugar de aplastarla a 224×224, sube validación a 0.242 y el top-5 a 0.589. Después se encontró que el color de las skins confunde al modelo: evaluar en grises mejora a un modelo entrenado a color, y entrenar y predecir solo en grises da la mejor versión en validación.

El modelo final es esa versión (v6): 0.316 de accuracy en validación, y 0.263 de accuracy, 0.526 de top-5 y 0.230 de F1 macro en test, contra 0.026 que daría el azar. Todas las versiones siguen llegando a ~99% en train, así que el límite está en la cantidad de imágenes por campeón.

La interfaz web usa la v8, que es la misma configuración de la v6 reentrenada con train + validación + test para aprovechar todas las imágenes. Como no queda ninguna imagen sin usar, la v8 no se puede medir: las métricas que se reportan son las de la v6. Detalles completos en el reporte.
