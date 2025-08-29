## Exploração dos Dados

Para o projeto foi utilizado o dataset [Spambase](https://archive.ics.uci.edu/dataset/94/spambase)

O dataset utilizado foi o Spambase, Ele contém 4601 observações, no qual são emails e apresenta 58 atributos, sendo 57 características para avaliar se o email é spam ou não e 1 alvo indicando spam ou não-spam. 

Para entendermos melhor os dados que estamos manipulando, foi gerado algumas analises baseado nos dados do projeto.


### Distribuição da Classe
![Distribuição Spam vs Não-Spam](distribuicao_alvo.png)

O dataset contém 4601, 2788 emails não-spam e 1813 emails spam.  
Apesar de não ser perfeitamente balanceado, ainda há uma boa representatividade das duas classes.

### Top 10 Palavras mais Frequentes
![Top 10 Palavras](top10_palavras.png)

As palavras mais comuns em emails incluem “you”, “your”, “free” e “our”.  
Palavras utilizadas com frequencia pelas pessoas e que mostra como spams apelam para comunicação direta com o usuário e ofertas atrativas.


### Caracteres Especiais por Classe
![Caracteres por Classe](caracteres_por_classe.png)

Emails classificados como spam apresentam maior frequência dos caracteres “!” e “$”, usados para chamar atenção (“OFERTA!!!”, “GANHE $$$”).  
Já os não-spam possuem esses símbolos em quantidade bem menor.


### Uso de Maiúsculas
![Maiúsculas por Classe](capslock_por_classe.png)

Os spams tendem a utilizar mais letras maiúsculas ao longo do texto, com picos muito acima dos emails normais.  
Isso reflete a prática de destacar trechos inteiros com maiúsculas para atrair a atenção do leitor.

## Pré-processamento

O dataset não apresentou valores nulos como testado no exploração.py e não foi necessário realizar normalização, 
uma vez que o algoritmo de árvore de decisão não depende de escalonamento.   


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

A partir da avaliação o modelo indica bom desempenho para esta base. 