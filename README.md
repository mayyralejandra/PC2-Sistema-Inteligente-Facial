# Evaluación 2 - Sistema Inteligente de Análisis Facial

Sistema de visión por computador desarrollado para la **Evaluación 2**, orientado a la detección de rostros, reconocimiento de identidad y clasificación de emociones mediante modelos de aprendizaje automático y aprendizaje profundo.

## 📁 Estructura del proyecto

```text
PC2_Sistema_Inteligente_Facial_FINAL/
│
├── dataset/
│   ├── P001/
│   ├── P002/
│   ├── P003/
│   ├── P004/
│   ├── P005/
│   └── metadata.csv
│
├── modelos/
│   └── identidad_insightface_svm.joblib
│
├── codigo/
│   ├── app.py
│   ├── entrenar_y_evaluar.py
│   └── baseline_local.py
│
├── notebooks/
│   ├── 01_preparacion_dataset.ipynb
│   └── 02_modelo_final_y_metricas.ipynb
│
├── originales_base/
│
├── resultados/
│   ├── detecciones_insightface.csv
│   ├── predicciones_emocion_deepface.csv
│   └── metricas_finales.json
│
├── requirements.txt
├── packages.txt
├── runtime.txt
├── README.md
└── Informe_Evaluacion_2_FINAL.docx
```

## 🎯 Objetivo

Desarrollar un sistema inteligente capaz de:

* Detectar rostros en imágenes.
* Reconocer la identidad de las personas registradas.
* Clasificar emociones faciales.
* Mostrar los resultados mediante una interfaz interactiva.
* Evaluar el desempeño de los modelos mediante métricas de clasificación.

## 🗂️ Dataset

El dataset contiene **50 imágenes correspondientes a 5 identidades, con 10 imágenes por persona**.

Cada identidad se encuentra organizada en una carpeta independiente:

```text
P001 → 10 imágenes
P002 → 10 imágenes
P003 → 10 imágenes
P004 → 10 imágenes
P005 → 10 imágenes
```

El archivo `dataset/metadata.csv` contiene la información asociada a cada imagen, incluyendo:

* Identificador de la persona.
* Nombre.
* Número de fotografía.
* Nombre del archivo.
* Etiqueta visual.
* Emoción sugerida.
* Ruta de la imagen.

Las imágenes incluyen variaciones de expresión facial y orientación del rostro.

> **Nota:** Las imágenes utilizadas para esta versión del proyecto corresponden a variaciones generadas a partir de cinco retratos base. Por ello, esta versión se utiliza principalmente para el desarrollo, evaluación y demostración del sistema. Si se requiere cumplir literalmente con una captura propia del grupo, las imágenes pueden ser reemplazadas manteniendo la misma estructura y metadatos.

## 🧠 Diseño del sistema

### 1. Detección facial

Se utiliza **OpenCV Haar Cascade** para localizar el rostro en las imágenes antes de generar los embeddings.

### 2. Reconocimiento de identidad

Para la representación facial se utiliza **InsightFace**, específicamente el modelo `buffalo_l`.

El proceso consiste en:

```text
Imagen
   ↓
Detección del rostro
   ↓
Recorte facial
   ↓
Embedding facial mediante InsightFace
   ↓
Normalización L2
   ↓
Clasificación SVM
   ↓
Identidad + confianza
```

El modelo entrenado se encuentra en:

```text
modelos/identidad_insightface_svm.joblib
```

### 3. Reconocimiento de emociones

La clasificación de emociones se realiza mediante **DeepFace** utilizando un modelo preentrenado.

Las emociones consideradas son:

* Feliz
* Triste
* Neutral
* Sorpresa
* Enojo

### 4. Información adicional

La interfaz también utiliza el modelo `genderage` disponible en InsightFace para mostrar:

* Sexo estimado.
* Edad estimada.

Estos valores corresponden a estimaciones del modelo y no deben interpretarse como información demográfica exacta.

## 🖥️ Interfaz interactiva

La interfaz fue desarrollada con **Streamlit**.

Permite:

* Cargar una imagen.
* Tomar una fotografía mediante cámara.
* Detectar el rostro.
* Identificar a la persona.
* Mostrar la confianza del reconocimiento.
* Clasificar la emoción.
* Mostrar la confianza de la emoción.
* Mostrar edad estimada.
* Mostrar sexo estimado.
* Visualizar el rostro detectado mediante un recuadro.

Para ejecutar la aplicación localmente:

```bash
python -m streamlit run codigo/app.py
```

## 📊 Evaluación del sistema

La evaluación del reconocimiento de identidad se realizó utilizando una división por persona:

* Imágenes 01–08: entrenamiento.
* Imágenes 09–10: prueba.

### Reconocimiento de identidad

Resultados obtenidos con InsightFace + SVM:

| Métrica         | Resultado |
| --------------- | --------: |
| Accuracy        |   88.89 % |
| Precision macro |   93.33 % |
| Recall macro    |   90.00 % |
| F1-score macro  |   89.33 % |

### Reconocimiento de emociones

Resultados obtenidos con DeepFace:

| Métrica         | Resultado |
| --------------- | --------: |
| Accuracy        |   52.00 % |
| Precision macro |   65.63 % |
| Recall macro    |   55.00 % |
| F1-score macro  |   59.50 % |

Los resultados completos y las matrices de confusión se encuentran en:

```text
resultados/metricas_finales.json
```

También se incluyen los archivos:

```text
resultados/detecciones_insightface.csv
resultados/predicciones_emocion_deepface.csv
```

## 🔬 Baselines de control

Como referencia se implementaron modelos baseline locales utilizando características HOG + SVM.

Resultados obtenidos:

| Modelo              | Accuracy | F1 macro |
| ------------------- | -------: | -------: |
| Identidad HOG + SVM | 100.00 % | 100.00 % |
| Emoción HOG + SVM   |  20.00 % |  13.00 % |

Estos resultados corresponden únicamente a modelos de control y **no representan el desempeño del modelo final** basado en InsightFace y DeepFace.

## ▶️ Ejecución del proyecto

### Opción 1: ejecución local

Se recomienda utilizar **Python 3.12**.

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar la interfaz:

```bash
python -m streamlit run codigo/app.py
```

### Opción 2: notebooks

Los notebooks contienen el flujo de preparación y ejecución utilizado durante el desarrollo:

```text
notebooks/01_preparacion_dataset.ipynb
notebooks/02_modelo_final_y_metricas.ipynb
```

El segundo notebook contiene el proceso de entrenamiento y evaluación del modelo final.

## 📦 Dependencias principales

* Python 3.12
* Streamlit
* OpenCV
* InsightFace
* DeepFace
* TensorFlow
* scikit-learn
* NumPy
* Pandas
* Pillow
* ONNX Runtime

Las versiones utilizadas se encuentran especificadas en:

```text
requirements.txt
```

Las dependencias adicionales del sistema Linux necesarias para la ejecución de OpenCV en Streamlit Cloud se encuentran en:

```text
packages.txt
```

## ☁️ Despliegue

La interfaz está preparada para ejecutarse en **Streamlit Community Cloud** a partir del repositorio de GitHub.

Archivo principal:

```text
codigo/app.py
```

La aplicación utiliza Python 3.12 para mantener compatibilidad con las dependencias de visión por computador utilizadas en el proyecto.

## ⚠️ Limitaciones

El desempeño del sistema puede variar dependiendo de:

* Iluminación.
* Orientación del rostro.
* Distancia respecto a la cámara.
* Oclusiones.
* Calidad de la imagen.
* Expresiones faciales.

Asimismo, las imágenes del dataset actual son variaciones generadas a partir de retratos base, por lo que una evaluación con fotografías reales capturadas bajo diferentes condiciones permitiría realizar una validación más representativa.

## 👩‍💻 Proyecto académico

**Evaluación 2 – Sistema Inteligente de Análisis Facial**

Componentes principales:

**Detección facial → InsightFace/OpenCV → Embeddings → SVM → Identidad**

**Detección facial → DeepFace → Emoción**

**Streamlit → Interfaz interactiva**

Proyecto desarrollado con fines académicos.
