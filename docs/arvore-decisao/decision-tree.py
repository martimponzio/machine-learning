import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

PATH_DATA = "source"  # onde estão spambase.csv e spambase.names

# Ler nomes das colunas do .names
colnames = []
with open(f"{PATH_DATA}/spambase.names", "r", encoding="utf-8", errors="ignore") as f:
    for ln in f:
        ln = ln.strip()
        if not ln or ln.startswith("|"):
            continue
        m = re.match(r"^([A-Za-z0-9_\.]+)\s*:", ln)
        if m:
            colnames.append(m.group(1))
colnames = colnames[:57] + ["is_spam"]  # 57 features + alvo

# Ler dados (sem header) e aplicar nomes
df = pd.read_csv(f"{PATH_DATA}/spambase.csv", header=None)
df.columns = colnames

# Separar X / y
X = df.drop(columns=["is_spam"])
y = df["is_spam"].astype(int)

# Split (igual ao prof; se quiser, pode pôr stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treinar e avaliar
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Plotar árvore (parcial) e salvar SVG
plt.figure(figsize=(12, 10))
tree.plot_tree(classifier, max_depth=3, filled=True)
plt.tight_layout()
plt.savefig("docs/arvore-decisao/tree.svg", format="svg")
plt.close()
