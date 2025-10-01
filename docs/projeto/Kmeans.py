# docs/projeto/exploracao_kmeans.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------
# Caminhos
# ---------------------------
PATH_DATA = "source/breast-cancer.csv"
SAIDA_GRAFICOS = "docs/projeto"
os.makedirs(SAIDA_GRAFICOS, exist_ok=True)

# ---------------------------
# Ler dados
# ---------------------------
df = pd.read_csv(PATH_DATA)

# Descobrir coluna alvo (se existir) e separar apenas colunas numéricas para X
cands = [c for c in df.columns if c.lower() in ("diagnosis", "target", "class", "label")]
if cands:
    alvo_col = cands[0]
    y_raw = df[alvo_col]
    X = df.drop(columns=[alvo_col]).select_dtypes(include="number").copy()
else:
    # Se não houver alvo, apenas use todas as numéricas para clusterização
    X = df.select_dtypes(include="number").copy()

# ---------------------------
# Padronização (necessária para K-Means)
# ---------------------------
scaler = StandardScaler()
X_escalonado = scaler.fit_transform(X)

# ---------------------------
# Gráfico 1: Método do Cotovelo (inércia)
# ---------------------------
Ks = range(2, 11)
inertias = []
sil_scores = []

for kk in Ks:
    km = KMeans(n_clusters=kk, random_state=42, n_init=10)
    rotulos = km.fit_predict(X_escalonado)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_escalonado, rotulos))

plt.figure(figsize=(6, 4))
plt.plot(list(Ks), inertias, marker="o")
plt.xticks(list(Ks))
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inércia (WCSS)")
plt.title("Método do Cotovelo — K-Means (Breast Cancer)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAIDA_GRAFICOS}/bc_kmeans_elbow.png", dpi=180)
plt.close()

# ---------------------------
# Gráfico 2: Coeficiente de Silhouette
# ---------------------------
plt.figure(figsize=(6, 4))
plt.plot(list(Ks), sil_scores, marker="o")
plt.xticks(list(Ks))
plt.xlabel("Número de clusters (k)")
plt.ylabel("Coeficiente de Silhouette (médio)")
plt.title("Coeficiente de Silhouette — K-Means (Breast Cancer)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAIDA_GRAFICOS}/bc_kmeans_silhouette.png", dpi=180)
plt.close()

# ---------------------------
# Gráfico 3: Dispersão 2D com centróides
#   (pega automaticamente as 2 features numéricas com maior variância)
# ---------------------------
variancias = X.var().sort_values(ascending=False)
feat_x, feat_y = variancias.index[:2]  # duas mais “espalhadas”

X2 = df[[feat_x, feat_y]].values

# padroniza só as 2 features para rodar o K-Means 2D
scaler_2d = StandardScaler()
X2_escal = scaler_2d.fit_transform(X2)

k2 = 2
kmeans_2d = KMeans(n_clusters=k2, random_state=42, n_init=10)
clusters_2d = kmeans_2d.fit_predict(X2_escal)

# volta centróides para a escala original das features
centroides_2d = scaler_2d.inverse_transform(kmeans_2d.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(X2[:, 0], X2[:, 1], c=clusters_2d, s=20, alpha=0.85)
plt.scatter(
    centroides_2d[:, 0], centroides_2d[:, 1],
    marker="*", s=250, c="red", edgecolors="black", label="Centróides"
)
plt.xlabel(feat_x)
plt.ylabel(feat_y)
plt.title("K-Means — Dispersão 2D com centróides (Breast Cancer)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAIDA_GRAFICOS}/bc_kmeans_scatter.png", dpi=180)
plt.close()

print("Gráficos salvos em docs/projeto/")
print(" - bc_kmeans_elbow.png")
print(" - bc_kmeans_silhouette.png")
print(" - bc_kmeans_scatter.png")
