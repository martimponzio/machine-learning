import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os

# Caminhos
PATH_DATA = "source/spambase.csv"
SAIDA_GRAFICOS = "docs/GRAFICOS_K-MEANS"
os.makedirs(SAIDA_GRAFICOS, exist_ok=True)

# Colunas
colnames = [
    "word_freq_make","word_freq_address","word_freq_all","word_freq_3d","word_freq_our",
    "word_freq_over","word_freq_remove","word_freq_internet","word_freq_order","word_freq_mail",
    "word_freq_receive","word_freq_will","word_freq_people","word_freq_report","word_freq_addresses",
    "word_freq_free","word_freq_business","word_freq_email","word_freq_you","word_freq_credit",
    "word_freq_your","word_freq_font","word_freq_000","word_freq_money","word_freq_hp",
    "word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab","word_freq_labs",
    "word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85",
    "word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct",
    "word_freq_cs","word_freq_meeting","word_freq_original","word_freq_project","word_freq_re",
    "word_freq_edu","word_freq_table","word_freq_conference",
    "char_freq_;","char_freq_(","char_freq_[","char_freq_!","char_freq_$","char_freq_#",
    "capital_run_length_average","capital_run_length_longest","capital_run_length_total",
    "is_spam"
]

# Ler dados
df = pd.read_csv(PATH_DATA, header=None, names=colnames)

X = df.drop(columns=["is_spam"])
y = df["is_spam"].astype(int)

# Padronização
scaler = StandardScaler()
X_escalonado = scaler.fit_transform(X)

# Método do elbow (inércia) e Silhouette (k=2..10)
Ks = range(2, 11)
inertias = []
sil_scores = []

for kk in Ks:
    km = KMeans(n_clusters=kk, random_state=42, n_init=10)
    labels = km.fit_predict(X_escalonado)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_escalonado, labels))

# Método do elbow
plt.figure(figsize=(6,4))
plt.plot(list(Ks), inertias, marker="o")
plt.xticks(list(Ks))
plt.xlabel("Número de clusters (k)")
plt.ylabel("Inércia (WCSS)")
plt.title("Método do elbow — K-Means")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAIDA_GRAFICOS}/GRAFICOS_K-MEANS_elbow.png", dpi=180)
plt.close()

# Silhouette
plt.figure(figsize=(6,4))
plt.plot(list(Ks), sil_scores, marker="o")
plt.xticks(list(Ks))
plt.xlabel("Número de clusters (k)")
plt.ylabel("Coeficiente de Silhouette (médio)")
plt.title("Coeficiente de Silhouette — K-Means")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAIDA_GRAFICOS}/GRAFICOS_K-MEANS_silhouette.png", dpi=180)
plt.close()




# Scatter 2D com centróides (usando 2 features reais)

feat_x = "char_freq_!"          # escolha 1
feat_y = "word_freq_free"       # escolha 2

# pega só essas duas colunas
X2 = df[[feat_x, feat_y]].values

# padroniza só essas duas (para o K-Means funcionar bem no 2D)
escalonador_2d = StandardScaler()
X2_escal = escalonador_2d.fit_transform(X2)

# roda K-Means em 2D
k2 = 2
kmeans_2d = KMeans(n_clusters=k2, random_state=42, n_init=10)
clusters_2d = kmeans_2d.fit_predict(X2_escal)

# padronizada --> traz de volta para a escala original 2D
centroides_2d = escalonador_2d.inverse_transform(kmeans_2d.cluster_centers_)

# plota
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X2[:, 0], X2[:, 1], c=clusters_2d, s=20, alpha=0.8)
plt.scatter(centroides_2d[:, 0], centroides_2d[:, 1],
            marker="*", s=250, c="red", edgecolors="black", label="Centróides")

plt.xlabel(feat_x)
plt.ylabel(feat_y)
plt.title("K-Means — Dispersão 2D com centróides")
plt.legend()
plt.tight_layout()
plt.savefig(f"{SAIDA_GRAFICOS}/GRAFICOS_K-MEANS_scatter.png", dpi=180)
plt.close()


print("Gráficos salvos em docs/GRAFICOS_K-MEANS/:")
print(" - GRAFICOS_K-MEANS_elbow.png")
print(" - GRAFICOS_K-MEANS_silhouette.png")
print(" - GRAFICOS_K-MEANS_scatter.png")
