import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PATH_DATA = Path("source")
PATH_OUT  = Path(".")
PATH_OUT.mkdir(parents=True, exist_ok=True)

# Tive que fazer isso, porque o arquivo para ler as colunas não estava lendo 58 e apenas 52.
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
df = pd.read_csv(PATH_DATA / "spambase.csv", header=None, names=colnames)

# Separação y e X
X = df.drop(columns=["is_spam"])
y = df["is_spam"].astype(int)

# Divisão treino/teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento do modelo e avaliação
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
print(f"Accuracy: {classifier.score(X_test, y_test):.2f}")

# Visualização da árvore de decisão
plt.figure(figsize=(12, 10))
tree.plot_tree(classifier, max_depth=3, filled=True)
plt.tight_layout()
plt.savefig(PATH_OUT / "tree.png", dpi=200)
plt.close()