from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# 1. Carregar base Spambase


PATH_DATA = Path("source")

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

df = pd.read_csv(PATH_DATA / "spambase.csv", header=None, names=colnames)

X = df.drop(columns="is_spam")
y = df["is_spam"]


# 2. Train / Test split + padronização


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Função auxiliar para gráficos


def gerar_graficos(y_true, y_pred, nome_modelo):

    # ---------- Heatmap da Matriz de Confusão ----------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["não spam", "spam"],
                yticklabels=["não spam", "spam"])
    plt.title(f"Matriz de Confusão - {nome_modelo}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.savefig(f"{nome_modelo}_confusion_matrix.png")
    plt.close()

    # ---------- Gráfico de Barras das Métricas ----------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metricas = [acc, prec, rec, f1]
    nomes = ["Acurácia", "Precisão", "Recall", "F1"]

    plt.figure(figsize=(7, 5))
    sns.barplot(x=nomes, y=metricas, palette="viridis")
    plt.ylim(0, 1)
    plt.title(f"Métricas de Desempenho - {nome_modelo}")
    plt.ylabel("Valor")
    for i, v in enumerate(metricas):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(f"{nome_modelo}_metricas.png")
    plt.close()




# 3. Função auxiliar para imprimir métricas + gerar gráficos


def imprimir_metricas(y_true, y_pred, nome_modelo: str) -> None:
    """
    Imprime métricas e gera imagens:
    - Heatmap matriz de confusão
    - Gráfico de barras métricas
    """

    # gráficos
    gerar_graficos(y_true, y_pred, nome_modelo)

    # métricas texto
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"MÉTRICAS - {nome_modelo}")
    print("=" * 60)
    print("Matriz de confusão (linhas = verdadeiro, colunas = predito):")
    print(cm)
    print("\nMétricas agregadas:")
    print(f"Acurácia : {acc:.4f}")
    print(f"Precisão : {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\nRelatório de classificação:")
    print(classification_report(y_true, y_pred, target_names=["não spam", "spam"]))

    print(f"\n⛭ IMAGENS GERADAS:\n- {nome_modelo}_confusion_matrix.png\n- {nome_modelo}_metricas.png")



# 4. KNN (modelo supervisionado)


def avaliar_knn(k: int = 5) -> None:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    imprimir_metricas(y_test, y_pred, f"KNN (k={k})")



# 5. K-Means (não supervisionado)


def avaliar_kmeans(n_clusters: int = 2) -> None:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    kmeans.fit(X_train_scaled)

    clusters_train = kmeans.predict(X_train_scaled)

    cluster_to_label = {}
    for c in range(n_clusters):
        mask = clusters_train == c
        majority_label = y_train[mask].mode()[0]
        cluster_to_label[c] = majority_label

    clusters_test = kmeans.predict(X_test_scaled)
    y_pred = pd.Series(clusters_test).map(cluster_to_label).to_numpy()

    imprimir_metricas(y_test, y_pred, "K-Means (clusters → rótulos)")



# 6. Execução principal


if __name__ == "__main__":
    avaliar_knn(k=5)
    avaliar_kmeans(n_clusters=2)
