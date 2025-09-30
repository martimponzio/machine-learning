
import os
import pandas as pd
import matplotlib.pyplot as plt

PATH_DATA = "source/breast-cancer.csv"
OUT_DIR   = "docs/projeto"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) Ler dados
df = pd.read_csv(PATH_DATA)

print("\n=== Dimensão do dataset ===")
print(df.shape)

print("\n=== Colunas ===")
print(list(df.columns))

print("\n=== Info (tipos) ===")
print(df.info())

print("\n=== Nulos por coluna ===")
print(df.isnull().sum())

# 2) Descobrir possível coluna-alvo (sem alterar nada)
# Preferência: 'diagnosis' ou 'target'; se não houver, usa a última coluna.
target_col = None
cands = [c for c in df.columns if c.lower() in ("diagnosis", "target", "class", "label")]
if cands:
    target_col = cands[0]
else:
    target_col = df.columns[-1]

print(f"\n>>> Coluna-alvo (suposição): {target_col}")

# 3) Distribuição do alvo (se for categórica/binária, só pra visualizar)
try:
    vc = df[target_col].value_counts()
    print("\n=== Distribuição do alvo ===")
    print(vc)
    ax = vc.plot(kind="bar", title=f"Distribuição do alvo: {target_col}")
    plt.xticks(rotation=0)
    plt.ylabel("Quantidade")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/bc_distribuicao_alvo.png", dpi=180)
    plt.close()
except Exception as e:
    print(f"Aviso: não consegui plotar distribuição do alvo ({e})")

# 4) Estatísticas descritivas de colunas numéricas
desc = df.select_dtypes(include="number").describe().T
print("\n=== Describe (numéricas) ===")
print(desc.head(10))

# salva uma amostra do describe em PNG simples (heatmapzinho com matplotlib puro)
num_cols = df.select_dtypes(include="number").columns.tolist()
# Hist de algumas colunas numéricas (as primeiras 6 para não poluir)
for col in num_cols[:6]:
    df[col].plot(kind="hist", bins=30, alpha=0.8, title=f"Histograma — {col}")
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/bc_hist_{col}.png", dpi=150)
    plt.close()

print(f"\nExploração pronta. Gráficos em {OUT_DIR}/")
