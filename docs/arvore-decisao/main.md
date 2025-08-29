## Exploração dos Dados

Para o projeto foi utilizado o dataset [Spambase](https://archive.ics.uci.edu/dataset/94/spambase)

O dataset utilizado foi o Spambase, Ele contém 4601 observações, no qual são emails e apresenta 58 atributos, sendo 57 características para avaliar se i email é spam ou não e 1 alvo indicando spam ou não-spam.

Desses 4601 observações chegamos que 2788 são não-spam e 1813 são spam (df["is_spam"].value_counts()).

## Pré-processamento

O dataset não apresentou valores nulos como testado no exploração.py e não foi necessário realizar normalização, 
uma vez que o algoritmo de árvore de decisão não depende de escalonamento.   

Além disso, algumas colunas foram renomeadas para facilitar interpretação, 
como por exemplo:
- `word_freq_free` para `freq_palavra_free`
- `word_freq_money` para `freq_palavra_dinheiro`
- `char_freq_!` para `freq_exclamacao`


``` python 
--8<-- "./docs/arvore-decisao/decision-tree.py"
```

## Divisão dos Dados

Os dados foram divididos em 80% treino e 20% teste utilizando train_test_split.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

## Treinamento do Modelo

Foi utilizada uma árvore de decisão (DecisionTreeClassifier) com random_state=42.

classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)


## Avaliação do Modelo

O modelo atingiu 92% de acurácia nos dados de teste.

## Conclusão