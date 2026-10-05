from pathlib import Path
import cv2, json, joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from deepface import DeepFace
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"
META = pd.read_csv(DATA / "metadata.csv")
MODELS = ROOT / "modelos"
RESULTS = ROOT / "resultados"
MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

# 1) Detector + embedding: InsightFace
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))

def embedding(path):
    img = cv2.imread(str(path))
    if img is None:
        return None

    # Detectar rostro con OpenCV
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return None

    # Seleccionar el rostro más grande
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    crop = img[y:y+h, x:x+w]

    # Generar embedding con ArcFace de InsightFace
    recognition = face_app.models["recognition"]
    feat = recognition.get_feat(crop)

    if feat is None or len(feat) == 0:
        return None

    emb = feat[0].astype(np.float32)

    # Normalización L2
    norm = np.linalg.norm(emb)
    if norm == 0:
        return None

    return emb / norm

# 2) Extraer embeddings de las 50 imágenes
X=[]; y=[]; rows=[]
for _, r in META.iterrows():
    p=DATA/r["ruta"]
    emb=embedding(p)
    ok=emb is not None
    rows.append({"archivo":r["archivo"],"id_persona":r["id_persona"],"detectado":ok})
    if ok:
        X.append(emb); y.append(r["id_persona"])
X=np.vstack(X); y=np.array(y)
pd.DataFrame(rows).to_csv(RESULTS/"detecciones_insightface.csv",index=False)

# 3) Split: fotos 1-8 train / 9-10 test por persona
valid = META.copy()
emb_map={row["archivo"]:embedding(DATA/row["ruta"]) for _,row in META.iterrows()}
trainX=[]; trainY=[]; testX=[]; testY=[]
for _,r in META.iterrows():
    e=emb_map[r["archivo"]]
    if e is None: continue
    if int(r["nro_foto"]) <= 8:
        trainX.append(e); trainY.append(r["id_persona"])
    else:
        testX.append(e); testY.append(r["id_persona"])
clf=SVC(kernel="linear",probability=True,class_weight="balanced",random_state=42)
clf.fit(np.asarray(trainX),trainY)
pred=clf.predict(np.asarray(testX))
acc=accuracy_score(testY,pred)
p,r,f,_=precision_recall_fscore_support(testY,pred,average="macro",zero_division=0)
joblib.dump(clf,MODELS/"identidad_insightface_svm.joblib")

# 4) Emociones: DeepFace preentrenado. No se entrena con estas 50 imágenes; se evalúa.
MAP={"happy":"Feliz","sad":"Triste","neutral":"Neutral","surprise":"Sorpresa","angry":"Enojo",
     "fear":"Otro","disgust":"Otro"}
em_true=[]; em_pred=[]; em_rows=[]
for _,r0 in META.iterrows():
    pth=DATA/r0["ruta"]
    try:
        res=DeepFace.analyze(img_path=str(pth),actions=["emotion"],enforce_detection=False,silent=True)
        if isinstance(res,list): res=res[0]
        raw=res["dominant_emotion"]
        pred_em=MAP.get(raw,"Otro")
        conf=float(res["emotion"].get(raw,0.0))
    except Exception as e:
        raw="error"; pred_em="Otro"; conf=0.0
    em_true.append(r0["emocion_sugerida"]); em_pred.append(pred_em)
    em_rows.append({"archivo":r0["archivo"],"real":r0["emocion_sugerida"],"pred":pred_em,"confianza":conf,"raw":raw})
pd.DataFrame(em_rows).to_csv(RESULTS/"predicciones_emocion_deepface.csv",index=False)
labels=["Feliz","Triste","Neutral","Sorpresa","Enojo"]
eacc=accuracy_score(em_true,em_pred)
ep,er,ef,_=precision_recall_fscore_support(em_true,em_pred,labels=labels,average="macro",zero_division=0)

out={
 "identidad_insightface_svm":{"accuracy":acc,"precision_macro":p,"recall_macro":r,"f1_macro":f,
     "matriz":confusion_matrix(testY,pred,labels=sorted(set(testY))).tolist()},
 "emocion_deepface":{"accuracy":eacc,"precision_macro":ep,"recall_macro":er,"f1_macro":ef,
     "matriz":confusion_matrix(em_true,em_pred,labels=labels).tolist()}
}
(RESULTS/"metricas_finales.json").write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(out,indent=2,ensure_ascii=False))
