# Mini-max-projeto
Projeto no âmbito Inteligência Articial

1. [O que é o Minimax?](#o-que-é-o-minimax)
2. [Como funciona?](#como-funciona)
3. [Como o algoritmo Minimax explora a árvore?](#como-o-algoritmo-minimax-explora-a-árvore)
4. [Vantagens vs Desvantagens](#vantagens-vs-desvantagens)
5. [Desempenho e Otimizações (Exemplo)](#desempenho-e-otimizações-exemplo)
6. [Considerações para Implementação no Xadrez](#considerações-para-implementação-no-xadrez)
7. [Pontos de Vista de Especialistas em Relação ao Minimax no Contexto do Xadrez](#pontos-de-vista-de-especialistas-em-relação-ao-minimax-no-contexto-do-xadrez)
8. [Melhorias](#melhorias)
9. [Conclusões](#conclusões)

## O que é o Minimax?

O Minimax é uma regra de decisão usada em teoria de decisão, teoria dos jogos, estatísticas e filosofia para minimizar a possível perda em um cenário de pior caso. É frequentemente utilizado em jogos de dois jogadores baseados em turnos, como xadrez ou jogo do galo, mas também pode ser aplicado a várias outras situações envolvendo tomada de decisões sob incerteza.

No contexto da teoria dos jogos, especialmente em jogos com informação perfeita, o minimax é uma estratégia que envolve minimizar a possível perda em um cenário de pior caso. Ele assume que o oponente sempre fará os melhores movimentos possíveis para maximizar sua vantagem e tem como objetivo tomar decisões que minimizem a perda máxima potencial.

## Como funciona?

Jogador Maximizador: O jogador que está a tentar maximizar a sua pontuação 
(ou utilidade) escolherá o movimento que leva à pontuação mais alta possível.

Jogador Minimizador: O oponente, que está a tentar minimizar a pontuação do 
jogador maximizador, escolherá o movimento que leva à pontuação mais baixa 
possível para o jogador maximizador.

Cada ‘nó’ representa uma possível jogada, e cada ‘linha’ representa um turno (um jogador após o outro). 

![image](https://github.com/Becas26/Mini-max-projeto/assets/102540581/fef8d595-c852-424a-91d2-eaba338edc27)

## Como o algoritmo Minimax explora a árvore?

O algoritmo minimax explora a árvore do jogo de forma recursiva, avaliando cada nó (que representa um estado do jogo) assumindo que ambos os jogadores jogam de forma otimizada. Ele calcula o melhor movimento para o jogador atual (maximizador) assumindo que o oponente (minimizador) também jogará de forma otimizada. 

![image](https://github.com/Becas26/Mini-max-projeto/assets/102540581/da5dc16f-2a0e-45da-b744-b5b39e9145e7)

## Vantagens vs Desvantagens

Vantagens: 
Garantia de Melhor Decisão: Se ambos os jogadores jogarem de forma ótima, o Minimax garantirá que a melhor decisão possível seja tomada em qualquer situação. 
Conceitualmente simples: O conceito por trás do Minimax é relativamente simples de entender. Ele é baseado na ideia de considerar todos os movimentos possíveis e escolher o melhor para um jogador ou o pior para o oponente. 
Amplamente aplicável: O Minimax não é limitado a jogos específicos; pode ser aplicado a uma variedade de problemas de tomada de decisão com estrutura semelhante a um jogo. 
 
 
 
Desvantagens: 
Complexidade Exponencial: A principal desvantagem do Minimax é que sua complexidade aumenta exponencialmente à medida que o número de possíveis movimentos aumenta. Isso torna o Minimax impraticável para jogos muito complexos, a menos que sejam implementadas técnicas de otimização como a poda alfa-beta. 
 
Não lida com incerteza: O Minimax não lida bem com situações onde há incerteza ou aleatoriedade envolvida, como em jogos de cartas onde as cartas são retiradas aleatoriamente. 

## Desempenho e Otimizações (Exemplo)

O algoritmo Minimax é frequentemente utilizado em jogos de tabuleiro como o xadrez para determinar a melhor jogada possível em uma determinada posição. No entanto, devido à complexidade do xadrez, a aplicação direta do Minimax sem otimizações seria impraticável, mesmo para computadores modernos 

## Considerações para Implementação no Xadrez

Avaliação do Tabuleiro: 
É preciso ter uma função de avaliação que atribui um valor numérico a uma dada posição no tabuleiro. Esta função avalia características como a vantagem material, controle do centro, mobilidade das peças, segurança do rei, entre outras.

Quanto mais precisa e eficaz for essa função, melhor o desempenho do algoritmo. Nós decidimos utilizar dois tipos de avaliação para comparação de performance: 
1.	Uma heurística baseada em :
a.	Valor das peças 
b.	Tabela de valores para peças por quadrado 
c.	Peões isolados e passados 
d.	Segurança doa Reia 
e.	Mobilidade 
f.	Torres em filas/colunas abertas 
g.	Bispos alinhados 
h.	Atividade das peças 
i.	Capturas disponíveis

![image](https://github.com/Becas26/Mini-max-projeto/assets/102540581/ac19ff07-fb57-477f-9b53-af744c3e47bb)
 
 
 
Jogo médio resultante da heurística + minimax 
 
 
2.	Uma rede neuronal com a seguinte estrutura: 
a.	Camada de entrada (8,8,1) 
b.	Primeiro Bloco Convolucional com Conexão Residual e 
Regularização L2 
i. Duas camadas convolucionais 3x3 com 64 filtros e ativação ReLU. ii. As saídas dessas camadas são somadas para formar uma conexão residual. 
iii. Após isso, é aplicada uma camada de pooling 2x2. 
c.	Segundo Bloco Convolucional com Conexão Residual e 
Regularização L2 (igual mas 128 filtros) 
d.	Camada de flattening 
e.	Bloco Totalmente Conectado com Conexão Residual, 
Regularização L2 e Dropout 
i. Duas camadas densas com 512 neurónios cada e ativação ReLU. ii. As camadas possuem dropout de 50% para prevenir overfitting. 
iii. Uma conexão residual é adicionada entre as duas camadas densas. 
f.	Camada de output 
 
 ![image](https://github.com/Becas26/Mini-max-projeto/assets/102540581/39c1e5f1-34fc-455e-b81c-04d0fc231620)
![image](https://github.com/Becas26/Mini-max-projeto/assets/102540581/009e3d70-f1c6-4196-b3f4-8c0ca9869f15)

 
Jogo médio com rede neuronal + minimax (treino de 1000 jogos) 
 
Ordenação de Jogadas: 
Em ambos os casos, optamos por ordenar as jogadas através de uma heurística simples e rápida, de modo a aumentar pruning e o desempenho do algoritmo. 
Esta heurística é baseada em: 
1.	Diferença material 
2.	Capturas

## Pontos de Vista de Especialistas em Relação ao Minimax no Contexto do Xadrez

Perspetiva Positiva 

Robustez Teórica:  Especialistas reconhecem que o Minimax é uma abordagem teoricamente sólida para a tomada de decisões em jogos. Garante uma escolha ótima de jogadas quando o jogo é totalmente explorado até o final, assumindo que ambos os jogadores jogam de forma ideal. 

Aplicabilidade Geral:  O Minimax não é específico para o xadrez; ele pode ser aplicado a uma variedade de jogos de tabuleiro. Sua versatilidade o torna uma ferramenta valiosa em diferentes contextos de jogos estratégicos. 

Base para Melhorias:  Embora o Minimax em sua forma básica possa ser impraticável para jogos complexos como o xadrez, ele serve como base para técnicas mais avançadas, como a poda alfa-beta e métodos de avaliação heurística. Estas técnicas melhoram a eficiência computacional do Minimax. 

Perspetiva Negativa 

Complexidade Exponencial: reconhecem que a complexidade do Minimax cresce exponencialmente com a profundidade da árvore de jogo. Para jogos complexos como o xadrez, explorar todas as possibilidades até o final é geralmente impraticável devido ao grande número de movimentos possíveis. 

Limitações em Jogos de Longa Duração: Em jogos que podem durar muito tempo, como o xadrez, é impossível para um computador explorar todas as possibilidades em um tempo razoável. Isso significa que o Minimax muitas vezes precisa ser limitado por uma profundidade máxima, levando a decisões subótimas. 

Necessidade de Heurísticas: O Minimax não é eficaz sem heurísticas adequadas para avaliar a posição do tabuleiro. Desenvolver heurísticas precisas que capturem a complexidade do xadrez é um desafio em si mesmo e pode afetar significativamente a qualidade das decisões tomadas pelo algoritmo. 

## Melhorias 

Podiamos tornar o nosso algoritmo minimax com heuristica melhor de diversas formas: 

1.	Melhorar a heuristica para ter em conta muito mais nuances e estratégias e fazer uma melhor avaliação
2.	Melhorar a mini-heuristica feita para ordenar as jogadas para maximizar pruning de modo a melhorar a performance do algoritmo
3.	Diminuir o espaço de procura utilizando uma função que remova movimentos que sejam provavelmente piores para reduzir drasticamente o número de jogadas a considerar pelo algoritmo. 
Por outro lado, também poderíamos melhorar a nossa neural network, por exemplo: 
1.	Aumentando a complexidade da rede, adicionando mais layers convolucionais 
2.	Utilizando melhores recompensas 
3.	Recorrendo à otimização de hiperparâmetros automatizada 
4.	Uma melhor automatização do decaimento do learning rate

## Conclusões 

Os resultados iniciais do minimax com heurística foram bastante bons, atingindo uma precisão que varia entre os 50-80%. No entanto os resultados iniciais com a neural network variaram entre os 0-10%. 
Depois de treinar em self-play durante 1000 jogos, a precisão aumentou consideravelmente, variando entre os 40-60%. 
Experimentando heurísticas mais simples, os resultados são bastante piores que ambos estes resultados, permitindo-nos concluir que a função de avaliação é extremamente importante para o bom funcionamento do minimax. A ordenação de jogadas também é muito importante para a performance deste algoritmo, sendo que com um bom critério de ordenação existe muito mais pruning. 
Verificamos também que utilizando uma heurística determinística podemos obter resultados muito bons imediatamente, mas utilizando uma neural network precisamos de bastante tempo de treino para igualar a performance. Xadrez é um jogo completo, pelo que em teoria uma estratégia que inclua uma heurística perfeita será muito melhor ou igual a uma estratégia que utiliza neural network para avaliação. Contudo, o xadrez é um jogo com um número de estados diferentes extremamente elevado, e as melhores heurísticas de hoje em dia têm, como demonstrado pelo whitepaper do AlphaZero e a performance de Leela-chess, obtido uma performance ligeiramente pior do que uma rede muito bem optimizada e treinada como evaluation policy. No nosso caso, a heurística não é muito ótima pelo que seria de esperar que a neural network eventualmente ultrapassasse o poder avaliativo da heurística, devido a ter modelado mais estratégias e nuances não consideradas na heurística.  



 


