# docs/KNN/Knn.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# garantir pasta de saída
os.makedirs("docs/KNN", exist_ok=True)

PATH_DATA = "source/spambase.csv"

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

df = pd.read_csv(PATH_DATA, header=None, names=colnames)

X = df.drop(columns=["is_spam"])
y = df["is_spam"].astype(int)

# 2) Divisão + normalização
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit só no treino
X_test_scaled  = scaler.transform(X_test)        # só transform no teste

# 3) KNN (k=5) + avaliação
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.3f}")
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# 4) Gráfico 1: Dispersão (2 features)
feat_x = "char_freq_!"
feat_y = "word_freq_free"

plt.figure(figsize=(7,6))
plt.scatter(df.loc[y==0, feat_x], df.loc[y==0, feat_y], s=16, label="Não-Spam", marker="o", alpha=0.7)
plt.scatter(df.loc[y==1, feat_x], df.loc[y==1, feat_y], s=16, label="Spam",     marker="x", alpha=0.7)
plt.xlabel(feat_x)
plt.ylabel(feat_y)
plt.title("Dispersão por classe")
plt.legend()
plt.tight_layout()
plt.savefig("docs/KNN/knn_scatter.png", dpi=180)
plt.close()

# 5) Gráfico 2: Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Matriz de Confusão — KNN (k=5)")
ax.set_xlabel("Previsto")
ax.set_ylabel("Real")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Não-Spam","Spam"]); ax.set_yticklabels(["Não-Spam","Spam"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("docs/KNN/knn_confusion_heatmap.png", dpi=180)
plt.close()

# 6) Gráfico 3: Acurácia vs k (1..25)
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
plt.title("KNN: Acurácia vs k")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/KNN/knn_k_vs_accuracy.png", dpi=180)
plt.close()

# 7) Gráfico 4: Fronteira de decisão no espaço 2D (PCA)
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d  = pca.transform(X_test_scaled)

# Treina um KNN só para a visualização em 2D
knn_2d = KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X_train_2d, y_train)

# Limites pelos percentis para evitar que outliers “estiquem” os eixos
x1_low, x1_high = np.percentile(X_train_2d[:, 0], [1, 99])
x2_low, x2_high = np.percentile(X_train_2d[:, 1], [1, 99])

# Uma margem pequena em volta
pad_x1 = 0.5 * (x1_high - x1_low)
pad_x2 = 0.5 * (x2_high - x2_low)
x_min, x_max = x1_low - pad_x1, x1_high + pad_x1
y_min, y_max = x2_low - pad_x2, x2_high + pad_x2

# Grade e predição para a fronteira
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 400),
    np.linspace(y_min, y_max, 400)
)
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5],
             colors=["#ff9999", "#99ccff"])  # fundo: 0 / 1
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train,
            s=15, alpha=0.7, label="Treino")
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test,
            s=25, edgecolors="k", alpha=0.9, label="Teste")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("KNN — Fronteira de Decisão (PCA 2D)")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("docs/KNN/knn_decision_boundary.png", dpi=180)
plt.close()

print("\nGráficos salvos em docs/KNN/:")
print(" - knn_scatter.png")
print(" - knn_confusion_heatmap.png")
print(" - knn_k_vs_accuracy.png")
print(" - knn_decision_boundary.png")
