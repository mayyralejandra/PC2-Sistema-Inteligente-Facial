# Evaluación 2 - Sistema Inteligente de Análisis Facial

## Contenido
- `dataset/`: 50 imágenes individuales (P001-P005, 10 por identidad) y `metadata.csv`.
- `originales_base/`: los 5 retratos base aportados en la conversación.
- `codigo/entrenar_y_evaluar.py`: pipeline final con InsightFace + SVM y DeepFace.
- `codigo/app.py`: interfaz Streamlit.
- `notebooks/`: preparación y ejecución en Google Colab.
- `resultados/`: validación local, matrices y hoja de contacto.
- `Informe_Evaluacion_2_FINAL.docx`: informe académico.

## Ejecución recomendada en Google Colab
1. Sube/descomprime esta carpeta en `/content/PC2_Sistema_Inteligente_Facial_FINAL`.
2. Abre `notebooks/01_preparacion_dataset.ipynb`.
3. Ejecuta `notebooks/02_modelo_final_y_metricas.ipynb`.
4. El segundo notebook instala dependencias y crea `modelos/identidad_insightface_svm.joblib` y `resultados/metricas_finales.json`.
5. Para la interfaz: `streamlit run codigo/app.py` (en local) o usa un túnel si ejecutas Streamlit desde Colab.

## Diseño del sistema
- Detección/embeddings: InsightFace (`buffalo_l`).
- Identidad: SVM lineal con probabilidades.
- Emociones: DeepFace preentrenado.
- Interfaz: Streamlit.
- Split de identidad: imágenes 01-08 entrenamiento, 09-10 prueba por persona.

## Validación local incluida
- Haar detectó rostro en 41/50 imágenes (82.0%).
- Baseline de identidad HOG+SVM: Accuracy 1.00, F1 macro 1.00.
- Baseline de emoción HOG+SVM: Accuracy 0.20, F1 macro 0.13.

Estas métricas son de control local y **no deben confundirse** con los resultados finales de InsightFace/DeepFace; esos se calculan al ejecutar el notebook 02.

## Nota importante sobre el dataset
Las 50 imágenes son variaciones sintéticas creadas a partir de cinco retratos base. Esto sirve para desarrollar y demostrar el sistema, pero la consigna original solicita un dataset propio mediante captura de imágenes. Si el docente exige cumplimiento literal, reemplaza las imágenes por fotografías capturadas por el grupo manteniendo los mismos nombres/carpetas.
