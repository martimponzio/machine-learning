# docs/projeto/kmeans_bc.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score

PATH_DATA = "source/breast-cancer.csv"
OUT_DIR   = "docs/projeto"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(PATH_DATA)

# tentar alvo só para comparar clusters x rótulos (não é obrigatório no K-Means)
cands = [c for c in df.columns if c.lower() in ("diagnosis", "target", "class", "label")]
target_col = cands[0] if cands else None

if target_col is not None:
    y_raw = df[target_col]
else:
    y_raw = None

X = df.drop(columns=[target_col]) if target_col else df.copy()
X = X.select_dtypes(include="number").copy()

# padroniza
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# k=2 (benign/malignant costuma ser binário)
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_s)

# mapeia cluster -> classe majoritária (se tivermos y)
if y_raw is not None:
    if y_raw.dtype == object:
        map_try = {"M":1,"B":0,"malignant":1,"benign":0}
        y = y_raw.map(lambda v: map_try.get(str(v).lower(), map_try.get(str(v), v)))
        if y.dtype == object:
            y = pd.factorize(y_raw)[0]
    else:
        y = y_raw.astype(int)

    mapped = np.zeros_like(labels)
    for c in range(k):
        mask = (labels == c)
        majority = y[mask].mode()[0]
        mapped[mask] = majority

    acc = accuracy_score(y, mapped)
    cm  = confusion_matrix(y, mapped)
    print(f"Acurácia (clusters vs rótulos): {acc:.3f}")
    print("Matriz de Confusão:\n", cm)

# Elbow e silhouette (k=2..10)
Ks = range(2, 11)
inertias, sils = [], []
for kk in Ks:
    km = KMeans(n_clusters=kk, random_state=42, n_init=10)
    lab = km.fit_predict(X_s)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X_s, lab))

plt.figure(figsize=(6,4))
plt.plot(list(Ks), inertias, marker="o")
plt.xticks(list(Ks))
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inércia (WCSS)")
plt.title("Método do cotovelo — K-Means (Breast Cancer)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/bc_kmeans_elbow.png", dpi=180)
plt.close()

plt.figure(figsize=(6,4))
plt.plot(list(Ks), sils, marker="o")
plt.xticks(list(Ks))
plt.xlabel("Número de clusters (k)")
plt.ylabel("Coeficiente de Silhouette (médio)")
plt.title("Silhouette — K-Means (Breast Cancer)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/bc_kmeans_silhouette.png", dpi=180)
plt.close()

# Dispersão 2D simples usando as duas colunas de maior variância
vari = X.var().sort_values(ascending=False)
feat_x, feat_y = vari.index[:2]
X2 = df[[feat_x, feat_y]].values
X2_s = StandardScaler().fit_transform(X2)
km2 = KMeans(n_clusters=2, random_state=42, n_init=10)
lab2 = km2.fit_predict(X2_s)
cent2 = km2.cluster_centers_
# volta centróide pra escala original
from sklearn.preprocessing import StandardScaler as SS2
ss2 = SS2().fit(X2)
cent2_orig = ss2.inverse_transform(cent2)

plt.figure(figsize=(7,5))
plt.scatter(X2[:,0], X2[:,1], c=lab2, s=18, alpha=0.85)
plt.scatter(cent2_orig[:,0], cent2_orig[:,1], marker="*", s=260, c="red",
            edgecolors="k", label="Centróides")
plt.xlabel(feat_x); plt.ylabel(feat_y)
plt.title("K-Means — Dispersão 2D (maior variância)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/bc_kmeans_scatter2d.png", dpi=180)
plt.close()
