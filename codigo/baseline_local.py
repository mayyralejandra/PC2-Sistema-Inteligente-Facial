# Validación local ligera (sin descargar pesos externos)
# Usa HOG + SVM para comprobar que el dataset y el split funcionan.
# NO reemplaza al modelo final InsightFace/DeepFace.
from pathlib import Path
import pandas as pd, numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'dataset'; df=pd.read_csv(D/'metadata.csv')
X=[]
for _,r in df.iterrows():
    a=np.array(Image.open(D/r['ruta']).convert('L').resize((128,128)))
    X.append(hog(a,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2)))
X=np.asarray(X); tr=df.nro_foto<=8; te=~tr
m=SVC(kernel='linear').fit(X[tr],df.loc[tr,'id_persona'])
print(classification_report(df.loc[te,'id_persona'],m.predict(X[te]),zero_division=0))
