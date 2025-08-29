import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

PATH_DATA = "source"
PATH_OUT  = "docs/arvore-decisao"


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
df = pd.read_csv(f"{PATH_DATA}/spambase.csv", header=None, names=colnames)

# exibe as 5 primeiras linhas do dataframe
print("Dimensão:", df.shape)

# informações sobre o dataframe
print(df.info())

# verifica valores nulos em cada coluna
print(df.isnull().sum())

# verifica valores NaN em cada coluna
print(df.isna().sum())

# total de nulos no dataset
print("Total de valores nulos:", df.isnull().sum().sum())


# Distribuição da variável alvo, descobrir quantos são spam e não-spam

df["is_spam"].value_counts().plot(kind="bar", title="Distribuição: Spam vs Não-Spam")
plt.xticks([0, 1], ["Não-Spam", "Spam"], rotation=0)
plt.ylabel("Quantidade")
plt.tight_layout()
plt.savefig(f"{PATH_OUT}/distribuicao_alvo.png")
plt.close()

# Resultado 2788 não-spam vs 1813 spam


# Análise das 10 palavras mais frequentes em emails

word_cols = [c for c in df.columns if c.startswith("word_freq_")]
df[word_cols].mean().sort_values(ascending=False).head(10)\
  .plot(kind="barh", title="Top 10 Palavras mais Frequentes")
plt.xlabel("Frequência média (%)")
plt.tight_layout()
plt.savefig(f"{PATH_OUT}/top10_palavras.png")
plt.close()

# Análise da frequência de caracteres especiais por classe (spam x não-spam)

char_cols = [c for c in df.columns if c.startswith("char_freq_")]
df.groupby("is_spam")[char_cols].mean().T.plot(kind="bar", figsize=(10,6))
plt.title("Frequência média de caracteres por classe")
plt.ylabel("Frequência média (%)")
plt.savefig("docs/arvore-decisao/caracteres_por_classe.png")
plt.close()

# Maiúsculas (boxplot comparando spam x não-spam)

df.boxplot(column="capital_run_length_total", by="is_spam", figsize=(6,6))
plt.title("Uso de LETRAS MAIÚSCULAS por classe")
plt.suptitle("")
plt.xlabel("Classe (0 = Não-Spam, 1 = Spam)")
plt.ylabel("Total de letras maiúsculas")
plt.savefig("docs/arvore-decisao/capslock_por_classe.png")
plt.close()