# Modelo de Random Forest
## Introdução

Após a implementação da Árvore de Decisão, o próximo passo foi aplicar o Random Forest, uma técnica de aprendizado supervisionado baseada no princípio do ensemble learning.
O método consiste em treinar diversas árvores de decisão independentes e combinar seus resultados, de modo que o voto majoritário determine a classificação final.

Essa abordagem reduz o risco de overfitting (quando o modelo “decorra” os dados de treino) e aumenta a precisão geral da previsão, tornando o modelo mais robusto e estável.

## Implementação do Modelo

O modelo foi implementado utilizando apenas a biblioteca scikit-learn, conforme a metodologia adotada em aula.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
import numpy as np

--> Carrega o dataset Spambase

PATH_DATA = Path("source")
df = pd.read_csv(PATH_DATA / "spambase.csv", header=None, names=colnames)

X = df.drop(columns=["is_spam"])
y = df["is_spam"].astype(int)

--> Divide os dados em treino e teste (80/20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

--> Cria e treina o modelo Random Forest

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train, y_train)

--> Avalia o modelo

pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")

--> Mostra as 10 features mais importantes

importances = rf.feature_importances_
feat_names = np.array(X.columns)
order = np.argsort(importances)[::-1][:10]

## Avaliação do Modelo

O modelo de Random Forest apresentou um excelente desempenho, alcançando uma acurácia de 92,4% nos dados de teste.
Esse valor indica que, a cada 100 e-mails analisados, aproximadamente 92 foram classificados corretamente como spam ou não-spam.

O resultado demonstra que a combinação de várias árvores aumenta significativamente a capacidade de generalização do modelo, tornando-o mais confiável do que uma única árvore de decisão.

Principais Variáveis (Importância das Features)

As dez variáveis mais relevantes identificadas pelo modelo foram:

Ranking	Feature	Importância
1	char_freq_$	0.1329
2	char_freq_!	0.1249
3	word_freq_remove	0.1024
4	word_freq_free	0.0787
5	word_freq_hp	0.0551
6	capital_run_length_average	0.0535
7	word_freq_your	0.0522
8	capital_run_length_longest	0.0505
9	capital_run_length_total	0.0424
10	word_freq_money	0.0366

Essas variáveis refletem com clareza o comportamento típico de e-mails de spam:

Símbolos monetários e de ênfase ($ e !) aparecem em excesso, buscando chamar a atenção do leitor.

Palavras-chave como “free”, “money” e “your” são amplamente utilizadas para promover ofertas enganosas.

Letras maiúsculas e longos blocos de texto em caixa alta (capital_run_length) reforçam o padrão apelativo característico de mensagens de spam.

## Conclusão

O modelo de Random Forest aplicado ao dataset Spambase apresentou ótima performance e forte capacidade de generalização, com acurácia de 92,4%.
A técnica demonstrou-se mais estável e precisa que uma árvore de decisão isolada, pois reduz o impacto de ruídos e outliers no conjunto de dados.

A análise das variáveis mais importantes confirma o comportamento esperado: spams utilizam linguagem apelativa, repetem símbolos de ênfase e abusam de termos relacionados a ganhos financeiros.

De forma geral, o Random Forest mostrou-se uma ferramenta eficiente e confiável para detecção automática de e-mails spam — equilibrando desempenho, interpretabilidade e robustez.