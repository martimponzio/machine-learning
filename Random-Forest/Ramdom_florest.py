
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
import numpy as np

# Caminho do dataset
PATH_DATA = Path("source")

# Nomes das colunas do Spambase (58 variáveis)
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

# Carrega o dataset Spambase
df = pd.read_csv(PATH_DATA / "spambase.csv", header=None, names=colnames)

# Define as features (X) e o alvo (y)
X = df.drop(columns=["is_spam"])
y = df["is_spam"].astype(int)

# Divide em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Inicializa e treina o modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=100,     # número de árvores
    max_depth=5,          # profundidade máxima
    max_features='sqrt',  # número de variáveis por divisão
    random_state=42
)
rf.fit(X_train, y_train)

# Faz previsões e avalia
pred = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")

# Mostra as 10 features mais importantes
importances = rf.feature_importances_
feat_names = np.array(X.columns)
order = np.argsort(importances)[::-1][:10]

print("\nTop 10 Features mais importantes:")
for rank, idx in enumerate(order, start=1):
    print(f"{rank:2d}. {feat_names[idx]:35s} {importances[idx]:.6f}")
