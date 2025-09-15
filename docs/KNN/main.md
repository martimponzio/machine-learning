## exploração

O dataset contém 4601 observações, no qual cada linha representa um email, descrito por 58 atributos. Desses, 57 são características extraídas do texto (frequência de palavras, caracteres especiais, uso de maiúsculas, etc.) e 1 é a variável alvo (is_spam), indicando se o email é spam (1) ou não-spam (0).

Assim como na etapa da árvore de decisão, foi realizada uma análise exploratória para compreender melhor o comportamento das classes e identificar os atributos mais relevantes.

## Dispersão por classe

![Dispersão](knn_scatter.png)

O gráfico de dispersão foi construído utilizando a frequência do caractere ! e a palavra “free”, ambos comuns em spams.
Percebe-se que os emails classificados como spam (X vermelho) tendem a concentrar maiores valores dessas variáveis, enquanto os não-spam (círculos azuis) aparecem em menor intensidade.






![Matriz de Confusão](knn_confusion_heatmap.png)










![Acurácia vs k](knn_k_vs_accuracy.png)

## Montagem do Roteiro

Os pontos "tarefas" são os passos que devem ser seguidos para a realização do roteiro. Eles devem ser claros e objetivos. Com evidências claras de que foram realizados.

### Tarefa 1

Instalando o MAAS:

<!-- termynal -->

``` bash
sudo snap install maas --channel=3.5/Stable
```


![Tela do Dashboard do MAAS](./maas.png)
/// caption
Dashboard do MAAS
///

Conforme ilustrado acima, a tela inicial do MAAS apresenta um dashboard com informações sobre o estado atual dos servidores gerenciados. O dashboard é composto por diversos painéis, cada um exibindo informações sobre um aspecto específico do ambiente gerenciado. Os painéis podem ser configurados e personalizados de acordo com as necessidades do usuário.

### Tarefa 2

## App



### Tarefa 1

### Tarefa 2

Exemplo de diagrama

```mermaid
architecture-beta
    group api(cloud)[API]

    service db(database)[Database] in api
    service disk1(disk)[Storage] in api
    service disk2(disk)[Storage] in api
    service server(server)[Server] in api

    db:L -- R:server
    disk1:T -- B:server
    disk2:T -- B:db
```

[Mermaid](https://mermaid.js.org/syntax/architecture.html){:target="_blank"}

## Questionário, Projeto ou Plano

Esse seção deve ser preenchida apenas se houver demanda do roteiro.

## Discussões

Quais as dificuldades encontradas? O que foi mais fácil? O que foi mais difícil?

## Conclusão

O que foi possível concluir com a realização do roteiro?
