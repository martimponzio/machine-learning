# Métricas

## Introdução

A avaliação de modelos de classificação não deve se limitar apenas à acurácia, pois diferentes erros têm impactos distintos dependendo do problema.
No caso do dataset Spambase, que tem como objetivo identificar se um e-mail é spam ou não-spam, métricas como precisão, recall e F1-score tornam-se fundamentais.

Isso ocorre porque:

classificar um spam como não-spam (falso negativo) é um erro crítico,

enquanto classificar um e-mail legítimo como spam (falso positivo) também gera impacto indesejado.

Dessa forma, a análise completa das métricas é essencial para entender como cada modelo se comporta diante desse tipo de desequilíbrio e quais deles apresentam resultados mais adequados para o problema.


## Implementação do Modelo

O processo de avaliação foi realizado da seguinte forma:

Carregamento do dataset Spambase e separação das variáveis independentes e da variável alvo.

Divisão dos dados em treinamento e teste, utilizando stratify, para manter a proporção original entre spam e não-spam.

Padronização dos dados com StandardScaler, fundamental para modelos baseados em distância.

Treinamento de dois modelos:

KNN (k = 5), um método supervisionado que utiliza vizinhos mais próximos para classificação.

K-Means, um método não-supervisionado, onde os clusters são posteriormente mapeados para rótulos reais com base na classe majoritária do conjunto de treino.

Cálculo das métricas através do arquivo Métricas.py, que produz:

- Matriz de confusão

- Acurácia

- Precisão

- Recall

- F1-score

Esse fluxo permite comparar diretamente o desempenho de um modelo supervisionado e de um não-supervisionado no mesmo problema.


## Avaliação do Modelo
Desempenho do KNN (k = 5)

A matriz de confusão do KNN mostra como o modelo se comporta ao diferenciar e-mails de spam e não-spam:

![KNN — Matriz de Confusão](KNN (k=5)_confusion_matrix.png)

Nela, observamos os acertos para cada classe (verdadeiros positivos e verdadeiros negativos) e também os erros (falsos positivos e falsos negativos). A distribuição dos valores indica que o modelo consegue acertar a maior parte dos spams ao mesmo tempo em que mantém um bom desempenho na identificação de mensagens legítimas.




As métricas agregadas de desempenho do KNN são resumidas no gráfico abaixo:

![KNN — Métricas de Desempenho](KNN (k=5)_metricas.png)

A partir desse gráfico, podemos destacar:

- **Acurácia elevada**, mostrando que o modelo acerta a maior parte das previsões.  
- **Precisão alta**, indicando que, quando o modelo classifica um e-mail como spam, ele geralmente está correto.  
- **Recall alto**, o que é especialmente importante nesse contexto, pois mostra que a maioria dos spams é de fato identificada.  
- **F1-score balanceado**, refletindo um bom equilíbrio entre precisão e recall.

As métricas mostram que o KNN consegue aprender padrões linguísticos presentes no spam — como alta frequência de palavras relacionadas a ofertas, urgência e elementos promocionais — e utilizá-los de forma eficaz para classificar novas mensagens.


Desempenho do K-Means

No caso do K-Means, o modelo não utiliza os rótulos durante o treinamento. Ele apenas agrupa os exemplos em dois clusters, que posteriormente são associados às classes spam e não-spam com base na classe mais frequente em cada grupo.

A matriz de confusão resultante é:

![K-Means — Matriz de Confusão](K-Means (clusters → rótulos)_confusion_matrix.png)

Essa matriz revela que, embora o K-Means consiga separar parte dos spams e não-spams, há uma quantidade maior de erros quando comparado ao KNN. Em especial, observa-se que alguns e-mails legítimos são agrupados junto com spams e vice-versa, o que aumenta tanto falsos positivos quanto falsos negativos.

As métricas agregadas do K-Means podem ser visualizadas no gráfico a seguir:

![K-Means — Métricas de Desempenho](K-Means (clusters → rótulos)_metricas.png)

De forma geral, nota-se que:

- **Acurácia é mais baixa** em relação ao KNN.  
- **Precisão e recall** são inferiores, indicando maior dificuldade em identificar corretamente os spams e em evitar classificar e-mails legítimos como spam.  
- **F1-score reduzido** evidencia o desequilíbrio entre acertos e erros.

Apesar disso, o desempenho não-supervisionado ainda evidencia que o dataset possui um padrão natural de separação.
E-mails com maior frequência de palavras como “free”, “credit”, “remove”, bem como uso excessivo de caracteres como “!”, tendem a formar um agrupamento consistente associado à classe spam.

## Conclusão

O modelo de K-Means aplicado ao dataset Spambase mostrou-se uma ferramenta útil para a exploração e identificação de padrões nos emails. As features, ou seja, as variáveis que descrevem cada mensagem, como a frequência da palavra “free” e o uso do caractere “!”, tiveram papel fundamental na formação dos clusters, os grupos criados automaticamente pelo algoritmo. Observou-se que os emails com maior presença desses elementos tendem a ser agrupados em um cluster associado ao spam, enquanto aqueles com baixa frequência permanecem no grupo de não-spam. Embora o K-Means não atinja o mesmo desempenho de modelos supervisionados, como a Árvore de Decisão ou o KNN, ele alcançou uma separação satisfatória das classes, evidenciando que as características linguísticas e de formatação são fortes indicadores de spam. Essa abordagem reforça o valor do K-Means como técnica exploratória, capaz de revelar padrões ocultos nos dados e oferecer uma visão inicial relevante, mesmo sem utilizar a variável alvo.