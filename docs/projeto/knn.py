# docs/projeto/knn_bc.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

PATH_DATA = "source/breast-cancer.csv"
OUT_DIR   = "docs/projeto"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(PATH_DATA)

cands = [c for c in df.columns if c.lower() in ("diagnosis", "target", "class", "label")]
target_col = cands[0] if cands else df.columns[-1]

y_raw = df[target_col]
X = df.drop(columns=[target_col]).select_dtypes(include="number").copy()

if y_raw.dtype == object:
    map_try = {"M":1, "B":0, "malignant":1, "benign":0}
    y = y_raw.map(lambda v: map_try.get(str(v).lower(), map_try.get(str(v), v)))
    if y.dtype == object:
        y = pd.factorize(y_raw)[0]
else:
    y = y_raw.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s, y_train)
y_pred = knn.predict(X_test_s)

print(f"Acurácia: {accuracy_score(y_test, y_pred):.3f}")
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Acurácia vs k
ks = range(1, 26)
accs = []
for k in ks:
    mdl = KNeighborsClassifier(n_neighbors=k)
    mdl.fit(X_train_s, y_train)
    accs.append(accuracy_score(y_test, mdl.predict(X_test_s)))

plt.figure(figsize=(7,4))
plt.plot(list(ks), accs, marker="o")
plt.xticks(list(ks))
plt.xlabel("k (n_neighbors)")
plt.ylabel("Acurácia no teste")
plt.title("KNN: Acurácia vs k — Breast Cancer")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/bc_knn_k_vs_acc.png", dpi=180)
plt.close()
