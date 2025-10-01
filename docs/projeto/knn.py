# docs/projeto/knn_bc.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------
# Paths (sem função, direto)
# ---------------------------
SCRIPT_DIR = Path(__file__).resolve().parent     # .../docs/projeto
ROOT_DIR   = SCRIPT_DIR.parents[1]               # raiz do repo
PATH_DATA  = ROOT_DIR / "source" / "breast-cancer.csv"
OUT_DIR    = SCRIPT_DIR                          # salva PNGs aqui

# ---------------------------
# Ler dados + descobrir alvo
# ---------------------------
df = pd.read_csv(PATH_DATA)

# tenta achar a coluna alvo comum; se não achar, usa a última
cands = [c for c in df.columns if c.lower() in ("diagnosis", "target", "class", "label")]
target_col = cands[0] if cands else df.columns[-1]
y_raw = df[target_col]

# só features numéricas
X = df.drop(columns=[target_col]).select_dtypes(include="number").copy()

# codifica alvo (M/B, malignant/benign → 1/0); fallback: factorize
if y_raw.dtype == object:
    mapa = {"m": 1, "b": 0, "malignant": 1, "benign": 0}
    y = y_raw.map(lambda v: mapa.get(str(v).lower(), v))
    if y.dtype == object:
        y = pd.factorize(y_raw)[0]
    y = pd.Series(y).astype(int)
else:
    y = y_raw.astype(int)

print(f"[INFO] X shape: {X.shape} | y shape: {y.shape} | alvo: {target_col}")

# ---------------------------
# Split + normalização
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit no treino
X_test_scaled  = scaler.transform(X_test)        # transform no teste

# ---------------------------
# KNN (k=5) + avaliação
# ---------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.3f}")
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# ---------------------------
# Gráfico 1: Dispersão (2 features reais de maior variância), por classe
# ---------------------------
variancias = X.var().sort_values(ascending=False)
feat_x, feat_y = variancias.index[:2] if len(variancias) >= 2 else (X.columns[0], X.columns[1])

plt.figure(figsize=(7,6))
plt.scatter(X.loc[y==0, feat_x], X.loc[y==0, feat_y], s=16, label="Classe 0", marker="o", alpha=0.7)
plt.scatter(X.loc[y==1, feat_x], X.loc[y==1, feat_y], s=16, label="Classe 1", marker="x", alpha=0.7)
plt.xlabel(feat_x)
plt.ylabel(feat_y)
plt.title("Dispersão por classe (2 features)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "bc_knn_scatter.png", dpi=180)
plt.close()

# ---------------------------
# Gráfico 2: Matriz de confusão (heatmap simples)
# ---------------------------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Matriz de Confusão — KNN (k=5)")
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Classe 0","Classe 1"]); ax.set_yticklabels(["Classe 0","Classe 1"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(OUT_DIR / "bc_knn_confusion_heatmap.png", dpi=180)
plt.close()

# ---------------------------
# Gráfico 3: Acurácia vs k (1..25)
# ---------------------------
ks = range(1, 26)
accs = []
for k in ks:
    mdl = KNeighborsClassifier(n_neighbors=k)
    mdl.fit(X_train_scaled, y_train)
    accs.append(accuracy_score(y_test, mdl.predict(X_test_scaled)))

plt.figure(figsize=(7,4))
plt.plot(list(ks), accs, marker="o")
plt.xticks(list(ks))
plt.xlabel("k (n_neighbors)")
plt.ylabel("Acurácia no teste")
plt.title("KNN: Acurácia vs k — Breast Cancer")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "bc_knn_k_vs_accuracy.png", dpi=180)
plt.close()

# ---------------------------
# Gráfico 4: Fronteira de decisão (PCA 2D)
# ---------------------------
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d  = pca.transform(X_test_scaled)

knn_2d = KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X_train_2d, y_train)

# limites pelos percentis (evita eixo esticado por outliers)
x1_low, x1_high = np.percentile(X_train_2d[:, 0], [1, 99])
x2_low, x2_high = np.percentile(X_train_2d[:, 1], [1, 99])
pad_x1 = 0.5 * (x1_high - x1_low)
pad_x2 = 0.5 * (x2_high - x2_low)
x_min, x_max = x1_low - pad_x1, x1_high + pad_x1
y_min, y_max = x2_low - pad_x2, x2_high + pad_x2

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5],
             colors=["#ff9999", "#99ccff"])
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, s=15, alpha=0.7, label="Treino")
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test,  s=25, edgecolors="k", alpha=0.9, label="Teste")
plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
plt.title("KNN — Fronteira de Decisão (PCA 2D)")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(OUT_DIR / "bc_knn_decision_boundary.png", dpi=180)
plt.close()

print("\nGráficos salvos em docs/projeto/:")
print(" - bc_knn_scatter.png")
print(" - bc_knn_confusion_heatmap.png")
print(" - bc_knn_k_vs_accuracy.png")
print(" - bc_knn_decision_boundary.png")
