import pandas as pd

df = pd.read_csv("source/spambase.csv", header=None)

# exibe as 5 primeiras linhas do dataframe
print("Dimensão:", df.shape)


print(df.info())

# verifica valores nulos em cada coluna
print(df.isnull().sum())

print(df.isna().sum())

# total de nulos no dataset
print("Total de valores nulos:", df.isnull().sum().sum())
