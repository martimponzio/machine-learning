import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

PATH_DATA = "source/breast-cancer.csv"
OUT_DIR   = "docs/projeto"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(PATH_DATA)

# Descobrir alvo
cands = [c for c in df.columns if c.lower() in ("diagnosis", "target", "class", "label")]
target_col = cands[0] if cands else df.columns[-1]

y_raw = df[target_col]
X = df.drop(columns=[target_col])

# Codificar y se for texto (M/B, malignant/benign etc.)
if y_raw.dtype == object:
    map_try = {"M":1, "B":0, "malignant":1, "benign":0}
    y = y_raw.map(lambda v: map_try.get(str(v).lower(), map_try.get(str(v), v)))
    if y.dtype == object:  # fallback: factorize
        y = pd.factorize(y_raw)[0]
else:
    y = y_raw.astype(int)

# Remover colunas não numéricas do X (árvore até aceita, mas mantemos simples)
X = X.select_dtypes(include="number").copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.3f}")
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Árvore resumida (profundidade 3) para visual
plt.figure(figsize=(14, 10))
plot_tree(clf, feature_names=X.columns, class_names=["0","1"], max_depth=3, filled=True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/bc_tree.png", dpi=180)
plt.close()
