from pathlib import Path
import tempfile
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import cv2

from PIL import Image

from deepface import DeepFace
from insightface.app import FaceAnalysis
from insightface.app.common import Face


# ============================================================
# RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "dataset"
MODELS = ROOT / "modelos"

MODEL_PATH = MODELS / "identidad_insightface_svm.joblib"
META_PATH = DATA / "metadata.csv"


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Sistema Inteligente de Análisis Facial",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: #0b0b0b;
    color: white;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

h1, h2, h3 {
    color: #ffffff;
}

div[data-testid="stMetric"] {
    background: #171717;
    border: 1px solid #333;
    border-top: 4px solid #F4C430;
    padding: 12px;
    border-radius: 10px;
}

div[data-testid="stMetricLabel"] {
    color: #cccccc;
}

div[data-testid="stMetricValue"] {
    color: #F4C430;
}

.resultado-box {
    background: #171717;
    border: 1px solid #333;
    border-left: 5px solid #F4C430;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}

.footer {
    text-align: center;
    color: #777777;
    font-size: 0.8rem;
    margin-top: 30px;
    padding-top: 15px;
    border-top: 1px solid #333;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MAPAS
# ============================================================

MAP_EMOTION = {
    "happy": "Feliz",
    "sad": "Triste",
    "neutral": "Neutral",
    "surprise": "Sorpresa",
    "angry": "Enojo",
    "fear": "Miedo",
    "disgust": "Disgusto",
}

GENDER_MAP = {
    0: "Mujer",
    1: "Hombre"
}


# ============================================================
# CARGAR MODELOS
# ============================================================

@st.cache_resource
def cargar_modelos():

    if not MODEL_PATH.exists():
        return None, None

    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    face_app.prepare(
        ctx_id=-1,
        det_size=(640, 640)
    )

    modelo = joblib.load(
        MODEL_PATH
    )

    return face_app, modelo


# ============================================================
# CARGAR METADATA
# ============================================================

@st.cache_data
def cargar_metadata():

    if not META_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(
        META_PATH
    )


face_app, modelo = cargar_modelos()

meta = cargar_metadata()


# ============================================================
# VALIDACIÓN
# ============================================================

if face_app is None or modelo is None:

    st.error(
        "No se encontró el modelo de identidad."
    )

    st.code(
        "modelos/identidad_insightface_svm.joblib"
    )

    st.stop()


# ============================================================
# FUNCIÓN: OBTENER EMBEDDING
# ============================================================

def obtener_embedding(img):

    if img is None:
        return None, None, None

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None, None, None

    x, y, w, h = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    face = Face(
        bbox=np.array(
            [x, y, x + w, y + h],
            dtype=np.float32
        )
    )

    crop = img[
        y:y + h,
        x:x + w
    ]

    recognition = face_app.models[
        "recognition"
    ]

    feat = recognition.get_feat(
        crop
    )

    if feat is None or len(feat) == 0:
        return None, None, None

    embedding = feat[0].astype(
        np.float32
    )

    norma = np.linalg.norm(
        embedding
    )

    if norma == 0:
        return None, None, None

    embedding = embedding / norma

    return embedding, face, (x, y, w, h)


# ============================================================
# FUNCIÓN: EDAD Y SEXO
# ============================================================

def obtener_edad_sexo(img, face):

    try:

        genderage = face_app.models[
            "genderage"
        ]

        gender, age = genderage.get(
            img,
            face
        )

        sexo = GENDER_MAP.get(
            int(gender),
            "No determinado"
        )

        edad = int(age)

        return edad, sexo

    except Exception:

        return None, "No determinado"


# ============================================================
# FUNCIÓN: EMOCIÓN
# ============================================================

def obtener_emocion(img):

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as tmp:

            cv2.imwrite(
                tmp.name,
                img
            )

            ruta_temporal = tmp.name

        resultado = DeepFace.analyze(
            ruta_temporal,
            actions=["emotion"],
            enforce_detection=False,
            silent=True
        )

        if isinstance(resultado, list):
            resultado = resultado[0]

        raw_emotion = resultado.get(
            "dominant_emotion",
            "unknown"
        )

        scores = resultado.get(
            "emotion",
            {}
        ) or {}

        confianza = float(
            scores.get(
                raw_emotion,
                0.0
            )
        )

        emocion = MAP_EMOTION.get(
            raw_emotion,
            raw_emotion.capitalize()
        )

        return emocion, confianza

    except Exception:

        return "No determinada", 0.0


# ============================================================
# FUNCIÓN: OBTENER NOMBRE
# ============================================================

def obtener_nombre(identidad):

    if meta.empty:
        return identidad

    if "nombre" in meta.columns:

        columna_nombre = "nombre"

    elif "nombre_completo" in meta.columns:

        columna_nombre = "nombre_completo"

    else:

        return identidad

    filas = meta[
        meta["id_persona"].astype(str)
        == str(identidad)
    ]

    if len(filas) == 0:

        return identidad

    return str(
        filas.iloc[0][columna_nombre]
    )


# ============================================================
# ENCABEZADO
# ============================================================

st.title(
    "🧠 Sistema Inteligente de Análisis Facial"
)

st.caption(
    "Reconocimiento de identidad, estimación de edad y sexo "
    "y análisis de emociones"
)


# ============================================================
# INFORMACIÓN
# ============================================================

st.info(
    "El sistema analiza una imagen facial y genera "
    "automáticamente la identidad, edad estimada, sexo "
    "estimado y emoción predominante."
)


# ============================================================
# ENTRADA
# ============================================================

col_a, col_b = st.columns(2)

with col_a:

    archivo = st.file_uploader(
        "📁 Subir una fotografía",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

with col_b:

    camara = st.camera_input(
        "📷 O tomar una fotografía"
    )


fuente = (
    camara
    if camara is not None
    else archivo
)


# ============================================================
# UMBRAL
# ============================================================

umbral = st.slider(
    "Umbral mínimo de confianza para reconocer "
    "una persona registrada",
    min_value=0.20,
    max_value=0.90,
    value=0.35,
    step=0.05,
)


# ============================================================
# ANÁLISIS
# ============================================================

if fuente is not None:

    imagen = Image.open(
        fuente
    ).convert("RGB")

    col1, col2 = st.columns(
        [1, 1.2]
    )

    # --------------------------------------------------------
    # IMAGEN ORIGINAL
    # --------------------------------------------------------

    with col1:

        st.subheader(
            "Imagen analizada"
        )

        st.image(
            imagen,
            use_container_width=True
        )

    try:

        with st.spinner(
            "Analizando rostro..."
        ):

            # ------------------------------------------------
            # CONVERTIR IMAGEN
            # ------------------------------------------------

            rgb = np.array(
                imagen
            )

            bgr = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR
            )

            # ------------------------------------------------
            # ROSTRO Y EMBEDDING
            # ------------------------------------------------

            embedding, face, bbox = obtener_embedding(
                bgr
            )

            if embedding is None:

                raise ValueError(
                    "No se detectó correctamente "
                    "un rostro en la imagen."
                )

            # ------------------------------------------------
            # IDENTIDAD
            # ------------------------------------------------

            probabilidades = modelo.predict_proba(
                [embedding]
            )[0]

            indice = int(
                np.argmax(
                    probabilidades
                )
            )

            identidad = str(
                modelo.classes_[indice]
            )

            confianza_identidad = float(
                probabilidades[indice]
            )

            # ------------------------------------------------
            # NOMBRE
            # ------------------------------------------------

            nombre = obtener_nombre(
                identidad
            )

            if confianza_identidad < umbral:

                nombre_mostrar = (
                    "Persona no registrada"
                )

            else:

                nombre_mostrar = nombre

            # ------------------------------------------------
            # EDAD Y SEXO
            # ------------------------------------------------

            edad, sexo = obtener_edad_sexo(
                bgr,
                face
            )

            # ------------------------------------------------
            # EMOCIÓN
            # ------------------------------------------------

            emocion, confianza_emocion = obtener_emocion(
                bgr
            )

        # ====================================================
        # RECUADRO DEL ROSTRO
        # ====================================================

        x, y, w, h = bbox

        imagen_resultado = bgr.copy()

        cv2.rectangle(
            imagen_resultado,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        imagen_resultado = cv2.cvtColor(
            imagen_resultado,
            cv2.COLOR_BGR2RGB
        )

        with col1:

            st.subheader(
                "Rostro detectado"
            )

            st.image(
                imagen_resultado,
                use_container_width=True
            )

        # ====================================================
        # RESULTADOS
        # ====================================================

        with col2:

            st.subheader(
                "Resultado del análisis"
            )

            if nombre_mostrar == "Persona no registrada":

                st.warning(
                    "Persona no registrada"
                )

            else:

                st.success(
                    f"Identidad reconocida: "
                    f"{nombre_mostrar}"
                )

            # ------------------------------------------------
            # MÉTRICAS
            # ------------------------------------------------

            m1, m2 = st.columns(2)

            with m1:

                st.metric(
                    "👤 Nombre",
                    nombre_mostrar
                )

            with m2:

                st.metric(
                    "🆔 ID",
                    identidad
                )

            m3, m4 = st.columns(2)

            with m3:

                st.metric(
                    "🎯 Confianza identidad",
                    f"{confianza_identidad:.1%}"
                )

            with m4:

                if edad is not None:

                    edad_texto = (
                        f"{edad} años"
                    )

                else:

                    edad_texto = (
                        "No disponible"
                    )

                st.metric(
                    "🎂 Edad estimada",
                    edad_texto
                )

            m5, m6 = st.columns(2)

            with m5:

                st.metric(
                    "⚥ Sexo estimado",
                    sexo
                )

            with m6:

                st.metric(
                    "😊 Emoción",
                    emocion
                )

            st.metric(
                "📊 Confianza emocional",
                f"{confianza_emocion:.1f}%"
            )

            # ------------------------------------------------
            # ADVERTENCIA
            # ------------------------------------------------

            if nombre_mostrar == "Persona no registrada":

                st.warning(
                    "La fotografía no superó el umbral "
                    "configurado para mostrar una identidad "
                    "registrada."
                )

    except Exception:

        st.error(
            "No fue posible analizar correctamente "
            "la imagen."
        )

        st.caption(
            "Prueba con una fotografía donde el rostro "
            "sea visible, frontal y tenga buena iluminación."
        )


# ============================================================
# NOTA METODOLÓGICA
# ============================================================

st.markdown(
    """
    <div class="resultado-box">

    <b>Nota:</b> La edad y el sexo corresponden a
    estimaciones realizadas por el modelo. Los resultados
    pueden variar según la iluminación, orientación,
    distancia y calidad de la fotografía.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.markdown(
    """
    <div class="footer">

    Sistema desarrollado para el proyecto académico PC2<br>
    Reconocimiento facial mediante aprendizaje automático
    e inteligencia artificial

    </div>
    """,
    unsafe_allow_html=True
)
