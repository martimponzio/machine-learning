# docs/projeto/kmeans_bc.py
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score

# =========================
# Paths robustos
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent          # .../docs/projeto
ROOT_DIR   = SCRIPT_DIR.parents[1]                     # .../
PATH_DATA  = ROOT_DIR / "source" / "breast-cancer.csv"
OUT_DIR    = SCRIPT_DIR                                # salva PNGs aqui (docs/projeto)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1) Ler dados + descobrir alvo
# =========================
df = pd.read_csv(PATH_DATA)

cands = [c for c in df.columns if c.lower() in ("diagnosis", "target", "class", "label")]
target_col = cands[0] if cands else df.columns[-1]

y_raw = df[target_col]
X = df.drop(columns=[target_col]).select_dtypes(include="number").copy()

# Codifica alvo se vier como texto (M/B, malignant/benign etc.)
if y_raw.dtype == object:
    mapa = {"m": 1, "b": 0, "malignant": 1, "benign": 0}
    y = y_raw.map(lambda v: mapa.get(str(v).lower(), v))
    if y.dtype == object:  # fallback
        y = pd.factorize(y_raw)[0]
    y = pd.Series(y).astype(int)
else:
    y = y_raw.astype(int)

# =========================
# 2) Padronização
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 3) K-Means (k=2) + “acurácia” por mapeamento de maioria
# =========================
k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Mapeia cada cluster para a classe majoritária
mapped = np.zeros_like(clusters)
for c in range(k):
    mask = (clusters == c)
    maioria = pd.Series(y[mask]).mode()[0]
    mapped[mask] = maioria

acc = accuracy_score(y, mapped)
cm  = confusion_matrix(y, mapped)

print(f"Acurácia (comparando clusters com o alvo): {acc:.3f}")
print("Matriz de Confusão:\n", cm)

# =========================
# 4) Elbow e Silhouette (k=2..10)
# =========================
Ks = range(2, 11)
inertias = []
sil_scores = []
for kk in Ks:
    km = KMeans(n_clusters=kk, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Elbow
plt.figure(figsize=(6,4))
plt.plot(list(Ks), inertias, marker="o")
plt.xticks(list(Ks))
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inércia (WCSS)")
plt.title("K-Means — Método do cotovelo (elbow)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "bc_kmeans_elbow.png", dpi=180)
plt.close()

# Silhouette
plt.figure(figsize=(6,4))
plt.plot(list(Ks), sil_scores, marker="o")
plt.xticks(list(Ks))
plt.xlabel("Número de clusters (k)")
plt.ylabel("Coeficiente de Silhouette (médio)")
plt.title("K-Means — Silhouette vs k")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "bc_kmeans_silhouette.png", dpi=180)
plt.close()

# =========================
# 5) Dispersão 2D com centróides
#    (usa duas features com maior variância)
# =========================
variancias = X.var().sort_values(ascending=False)
feat_x, feat_y = variancias.index[:2]

X2 = X[[feat_x, feat_y]].values
sc2 = StandardScaler()
X2_scaled = sc2.fit_transform(X2)

kmeans_2d = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_2d = kmeans_2d.fit_predict(X2_scaled)
centros_2d = sc2.inverse_transform(kmeans_2d.cluster_centers_)

plt.figure(figsize=(8,6))
plt.scatter(X2[:,0], X2[:,1], c=labels_2d, s=20, alpha=0.85)
plt.scatter(centros_2d[:,0], centros_2d[:,1],
            marker="*", s=250, c="red", edgecolors="black", label="Centróides")
plt.xlabel(feat_x)
plt.ylabel(feat_y)
plt.title("K-Means — Dispersão 2D com centróides (Breast Cancer)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "bc_kmeans_scatter.png", dpi=180)
plt.close()

print("\nGráficos salvos em docs/projeto/:")
print(" - bc_kmeans_elbow.png")
print(" - bc_kmeans_silhouette.png")
print(" - bc_kmeans_scatter.png")
