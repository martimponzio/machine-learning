# docs/KMEANS/kmeans.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, silhouette_score


CAMINHO_DADOS = "source/spambase.csv"
DIRETORIO_SAIDA = "docs/KMEANS"
os.makedirs(DIRETORIO_SAIDA, exist_ok=True)


nomes_colunas = [
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
dados = pd.read_csv(CAMINHO_DADOS, header=None, names=nomes_colunas)

X = dados.drop(columns=["is_spam"])
y = dados["is_spam"].astype(int)


#  Padronização

escalonador = StandardScaler()
X_escalonado = escalonador.fit_transform(X)


# KMeans com k=2

k = 2
kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)
clusters = kmeans.fit_predict(X_escalonado)


# Diagnóstico inicial

print("\nCrosstab is_spam x cluster (todas as features):")
print(pd.crosstab(y, clusters, rownames=['is_spam'], colnames=['cluster']))

rotulos_map = np.zeros_like(clusters)
for c in range(k):
    mascara = (clusters == c)
    maioria = y[mascara].mode()[0]
    rotulos_map[mascara] = maioria

acuracia = accuracy_score(y, rotulos_map)
matriz_confusao = confusion_matrix(y, rotulos_map)

print(f"\nAcurácia (todas features): {acuracia:.3f}")
print("Matriz de Confusão:\n", matriz_confusao)


# Teste com subset de features mais relevantes

cols_boas = [
    "word_freq_free","word_freq_money","word_freq_you","word_freq_your",
    "char_freq_!","char_freq_$",
    "capital_run_length_average","capital_run_length_longest","capital_run_length_total"
]
X_sub = dados[cols_boas]
X_escalonado_sub = StandardScaler().fit_transform(X_sub)

kmeans_sub = KMeans(n_clusters=2, random_state=42, n_init=50)
clusters_sub = kmeans_sub.fit_predict(X_escalonado_sub)

print("\nCrosstab is_spam x cluster (subset de features):")
print(pd.crosstab(y, clusters_sub, rownames=['is_spam'], colnames=['cluster']))

rotulos_map_sub = np.zeros_like(clusters_sub)
for c in range(2):
    mascara = (clusters_sub == c)
    maioria = y[mascara].mode()[0]
    rotulos_map_sub[mascara] = maioria

acuracia_sub = accuracy_score(y, rotulos_map_sub)
matriz_confusao_sub = confusion_matrix(y, rotulos_map_sub)

print(f"\nAcurácia (subset features): {acuracia_sub:.3f}")
print("Matriz de Confusão:\n", matriz_confusao_sub)

# (medida intrínseca)
#O Silhouette Score é uma métrica usada para avaliar qualidade de clusterização sem precisar usar os rótulos reais (is_spam).
#Ou seja, ele mede o quão bem separados e coesos estão os clusters que o K-Means formou.

sil = silhouette_score(X_escalonado_sub, clusters_sub)
print(f"\nSilhouette Score (subset features): {sil:.3f}")
