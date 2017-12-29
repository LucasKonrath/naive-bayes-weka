## Machine Learning com o Weka
### Veja como é surpreendentemente simples utilizar esta ferramenta.

O campo de Machine Learning pode ser um tanto obscuro para quem deseja iniciar seu aprendizado nele, porém meu objetivo neste artigo é mostrar o quão fácil pode ser iniciar o aprendizado utilizando o Weka. Primeiramente, o que é o Weka? O Weka é uma coleção de algoritmos de Machine Learning e Data Mining escrita em Java na Universidade de Waikato, Nova Zelândia. Por ser um arquivo .jar, utilizá-lo em código Java torna-se trivial, porém neste exemplo usaremos sua GUI pela facilidade e didática.

Este artigo será um passo a passo de como utilizar o Weka para, através de um algoritmo Naive Bayes, classificar um animal de acordo com suas características em 7 diferentes Classes de animais.

Caso você não saiba do que se trata Naive Bayes, aqui vai uma rápida explicação:

O Algoritmo Naive Bayes se trata de um algoritmo de Classificação (existem outros tipos no campo da ciência de dados, como algoritmos de clustering e associação, porém não irei entrar em detalhes neste artigo). Neste caso, é uma Classificação Supervisionada, pois o _dataset_ que iremos utilizar teve um _agente supervisor_, ou seja, alguma pessoa que analisou os dados e os classificou de acordo com suas características manualmente.
Tal algoritmo funciona bem no nosso caso, porém ele supõe que todos os elementos preditores sejam independentes entre si, o que normalmente não é o caso na vida real (normalmente temos atributos correlacionados).

Suponhamos que tenhamos o dataset abaixo, um exemplo clássico, que elenca várias opções e classifica se a pessoa irá ou não jogar golf no dia.

| Clima | Temperatura | Umidade | Vento | Jogar? |
| ----- |:----------: | :-----: | :---: | -----: |
| Ensolarado| Quente |Alta|   Fraco        |    Não |
|  Ensolarado     |   Quente          |  Alta       |  Forte     |  Não      |
|  Nublado     |  Quente           |  Alta       |   Fraco   |  Sim      |
|  Chuva     |    Moderada   |  Alta       |  Fraco     |   Sim     |
|  Chuva     |     Fria        |    Normal     | Fraco      |  Sim      |
| Chuva      |     Fria        |    Normal     |     Forte  |   Não      |
|  Nublado     |   Fria          |   Normal      |   Forte    |   Sim     |
|  Ensolarado     |    Moderada         |   Alta      | Fraco      |    Não    |
|  Ensolarado     |    Fria         |    Normal     |   Fraco    |   Sim     |
|  Chuva     |    Moderada         |   Normal      |    Forte   |   Sim     |
|  Ensolarado     |    Moderada   | Normal        |   Forte    |  Sim      |
|   Nublado    |    Moderada         |   Alta      |   Forte    |   Sim     |
|  Nublado     |   Quente          |    Normal     |   Fraco    |   Sim     |
|   Chuva    |    Moderado         |  Alta       |  Forte     |   Não     |

Notem que nosso dataset tem quatro atributos que geram a _classe Jogar_. O que nós precisamos fazer é montar uma tabela para cada um desses atributos, obtendo a porcentagem de chance da classe Jogar ser Sim ou Não baseada no atributo.
É importante frisar que existem 5 casos com o valor _Não_ e 9 com o valor _Sim_.

| Clima | Jogar = Sim | Jogar = Não | Total |
|-------|-------------|-------------|-------|
| Ensolarado | 2/9      | 3/5         | 5/14  |
| Nublado | 4/9 | 0/5 | 4/14 |
| Chuva | 3/9 | 2/5 | 5/14 |


| Temperatura | Jogar = Sim | Jogar = Não | Total |
|------------|-------------|--------------|--------|
|Quente | 2/9 | 2/5 | 4/14 |
|Moderada| 4/9 | 2/5 | 6/14 |
|Fria| 3/9 | 1/5 | 4/14 |

|Umidade| Jogar = Sim | Jogar = Não | Total|
|-------|-------------|------------|-------|
|Alta|3/9|4/5|7/14|
|Normal|6/9|1/5|7/14|

|Vento| Jogar = Sim | Jogar = Não | Total |
|-----| ------------| ----------- | ----- |
| Forte | 3/9 | 3/5 | 6/14 |
| Fraco | 6/9 | 2/5 | 8/14 |

Certo, agora sabemos a probabilidade da pessoa Jogar ou não, com base em cada um dos atributos isoladamente. Caso recebamos, digamos a seguinte instância:

(Clima = Ensolarado, Temperatura = Fria, Umidade = Alta, Vento = Forte)

Primeiro, veremos a probabilidade de Jogar ser sim:

Prob. (Clima = Ensolarado | Jogar = Sim) -> 2/9


Prob. (Temperatura = Fria | Jogar = Sim) -> 3/9


Prob. (Umidade = Alta | Jogar = Sim) -> 3/9


Prob. (Vento = Forte | Jogar = Sim) -> 3/9


Prob. (Jogar = Sim) -> 9/14

Então, multiplicaremos esses valores:

Prob.(X | Jogar = Sim) * Prob. (Jogar = Sim)

(2/9 * 3/9 * 3/9 * 3/9) * (9/14) = 0.0053

E faremos o mesmo para a probabilidade de não jogar.

Prob.(X | Jogar = Não) * Prob.(Jogar = Não)

(3/5 * 1/5 * 4/5 * 3/5) *  (5/14) = 0.0206

Agora o que nós devemos fazer é dividir estes dois valores pelo que chamamos de _evidência_, obtido pela multiplicação dos campos _Total_ de cada uma das tabelas de Atributos com os valores respectivos.

Evid. = Prob.(Clima = Ensolarado) * Prob.(Temperatura = Fria) * Prob.(Umidade = Alta) * Prob.(Vento = Forte)

Evid. = (5/14) * (4/14) * (7/14) * (6/14) = 0.02186

Então, dividindo os valores:

Prob. (Jogar = Sim | X) = 0.0053 / 0.02186 = 0.2424

Prob. (Jogar = Não | X) = 0.0206 / 0.02186 = 0.9421

Calculando as probabilidades, o algoritmo aponta uma chance de 79,54% de Não Jogar, e 20,46 % de Jogar. Ou seja, para esta instância, a classe teria o valor NÃO.

Ainda existe o conceito da Estimativa de Laplace, utilizado quando existem valores sem nenhuma ocorrência, empregado para resolver o problema de frequência zero no cálculo, porém irei falar sobre isso depois.

### Utilizando o Weka

Que emoção, vamos finalmente pôr a mão na massa.

1º Passo - Baixe e instale o Weka (link e instruções de instalação [aqui](https://www.cs.waikato.ac.nz/ml/weka/downloading.html)).

2º Passo - Baixe o dataset dos Animais [aqui](http://tunedit.org/repo/UCI/zoo.arff).

Certo, primeiro de tudo, abra esse arquivo .arff em um editor de texto, e vamos analisá-lo.

No início há muitos comentários explicando definições sobre o dataset, passando uma overview dos atributos e etc. Vou focar mais na parte técnica nas explicações.

`@RELATION zoo` 

Este atributo simplesmente define o nome do dataset, normalmente é igual ao nome do arquivo.

```
@ATTRIBUTE animal {aardvark,antelope,bass,bear,boar,buffalo,calf,carp,catfish,cavy,cheetah,chicken,chub,clam,crab,crayfish,crow,deer,dogfish,dolphin,dove,duck,elephant,flamingo,flea,frog,fruitbat,giraffe,girl,gnat,goat,gorilla,gull,haddock,hamster,hare,hawk,herring,honeybee,housefly,kiwi,ladybird,lark,leopard,lion,lobster,lynx,mink,mole,mongoose,moth,newt,octopus,opossum,oryx,ostrich,parakeet,penguin,pheasant,pike,piranha,pitviper,platypus,polecat,pony,porpoise,puma,pussycat,raccoon,reindeer,rhea,scorpion,seahorse,seal,sealion,seasnake,seawasp,skimmer,skua,slowworm,slug,sole,sparrow,squirrel,starfish,stingray,swan,termite,toad,tortoise,tuatara,tuna,vampire,vole,vulture,wallaby,wasp,wolf,worm,wren}
@ATTRIBUTE hair {false, true}
@ATTRIBUTE feathers {false, true}
@ATTRIBUTE eggs {false, true}
@ATTRIBUTE milk {false, true}
@ATTRIBUTE airborne {false, true}
@ATTRIBUTE aquatic {false, true}
@ATTRIBUTE predator {false, true}
@ATTRIBUTE toothed {false, true}
@ATTRIBUTE backbone {false, true}
@ATTRIBUTE breathes {false, true}
@ATTRIBUTE venomous {false, true}
@ATTRIBUTE fins {false, true}
@ATTRIBUTE legs INTEGER [0,9]
@ATTRIBUTE tail {false, true}
@ATTRIBUTE domestic {false, true}
@ATTRIBUTE catsize {false, true}
@ATTRIBUTE type { mammal, bird, reptile, fish, amphibian, insect, invertebrate }
```

Aqui são definidos os Atributos de cada Instância no Dataset, e seus possíveis valores. Perceba que animal é um dentre varios nomes de animais, hair é um boolean e legs é um Integer, ou seja, cada atributo tem um tipo específico.
Por padrão, o último atributo definido é considerado o atributo de classe (o atributo a partir do qual será gerado o modelo, bem como o que o modelo gerado irá calcular, neste caso, o type, escolhendo um dentre os 7 valores).

```
@DATA
%
% Instances (101):
%
aardvark,true,false,false,true,false,false,true,true,true,true,false,false,4,false,false,true,mammal
```

A partir daqui, são inseridos os valores de instâncias do dataset, com os atributos sendo passados na mesma ordem que foram definidos.
Caso um dos atributos não tenha valor definido, você passa ? no lugar dele. Veremos isto depois, quando formos calcular o type de uma instância.
Ou seja, caso não soubessemos que um _aardvark_ fosse um mamífero, passariamos a seguinte instancia para nosso modelo gerado calcular: 
```
aardvark,true,false,false,true,false,false,true,true,true,true,false,false,4,false,false,true,?
```

OK, agora que você já entende como funcionam os datasets no Weka, abra o programa e clique no botão Explorer.

O que faremos agora é clicar no botão Open File, e selecionar o arquivo zoo.arff que você baixou anteriormente.

Você deve estar com uma vista assim. Nesta aba, nós conseguimos visualizar as instâncias do dataset e seus atributos, filtrando graficamente por atributo. Nesta imagem podemos ver, por exemplo, que existe apenas uma instância de cada animal, exceto pelo sapo, que tem duas instâncias no dataset.

![](weka1.png)

Selecionando o atributo Milk, por exemplo, podemos ver que todas as instâncias em que ele é verdadeiro foram classificadas como Mamíferos. Faz sentido, não?

![](weka2.png)

Selecionando o atributo Type, podemos ver quantas instâncias pertencem a cada tipo. 

![](weka3.png)

Certo, o que faremos agora é gerar o modelo utilizando o algoritmo Naive Bayes. Clique na aba Classify (lembre-se, o Naive Bayes é um algoritmo de classificação). Clique na opção Choose e selecione o NaiveBayes, conforme imagem.

![](weka4.png)

Em Test Options, selecione a opção Use training set. O que isto vai fazer é utilizar o arquivo arrf que você carregou anteriormente para treinar o modelo de dados. Clique em Start.

O output imprimiu muita coisa interessante, não?
Vamos repassá-las uma a uma.

Essa é aquela matriz que tinhamos montado antes, para o atributo animal. 

![](weka5.png)

Ué, mas como assim, existe 1 animal desses em todas as classes? Isso se refere à estimativa de Laplace, que falei brevemente anteriormente.
Porque isto ocorre? Caso uma combinação não existe na base, sua porcentagem sendo 0/X, por exemplo, ao realizar a multiplicação para calcular a probabilidade de ocorrência dela, seria 0. Basicamente, quando houver uma combinação instância/classe com 0 ocorrências, é adicionada uma combinação fictícia simulando-a. A idéia é que a porcentagem, então, nunca fique zerada. Isso deve ser feito em todas as classes para evitar viés nos cálculos (o que explica porque a classificação correta do animal tem valor 2.0, e no caso do Sapo, 3.0).

Agora, temos nosso modelo construído e que será testado. 

![](weka6.png)

Na parte de cima, temos o relatório do nosso dataset original submetido ao modelo gerado pelo algoritmo Naive Bayes. _Correctly Classified Instances_ é a quantidade de instâncias que, submetidas ao modelo gerado, foram classificadas igual à sua classificação no dataset original. Neste caso, obtivemos 100% de sucesso, pois é um dataset relativamente extenso e completo, com muitos atributos e instâncias.

A matriz de confusão deve ser lida assim:
à direita está a classe original da instância, conforme classificada no dataset inicial. Na parte de cima está a classe definida pelo algoritmo Naive Bayes gerado. Ou seja, qualquer desvio da diagonal principal significa um erro.

Estamos quase lá, falta exportar o nosso modelo gerado e testá-lo com um animal que quisermos. 

Selecione o ultimo result do seu modelo na aba result list, clique com o mouse direito e escolha a opção save model. Escolha um nome para o arquivo e salve-o como um arquivo .model.
![](weka7.png)

Certo, já temos o modelo gerado, falta escolher um animal e vamos testá-lo agora. 

![](canguru-e-filhote.jpg)
Canguru, eu escolho você!

Vamos criar nosso próprio arrf agora para ver como o modelo gerado se comporta. Crie um arquivo novo copiando o texto deste.

```
@RELATION zoo

@ATTRIBUTE animal {kangaroo}
@ATTRIBUTE hair {false, true}
@ATTRIBUTE feathers {false, true}
@ATTRIBUTE eggs {false, true}
@ATTRIBUTE milk {false, true}
@ATTRIBUTE airborne {false, true}
@ATTRIBUTE aquatic {false, true}
@ATTRIBUTE predator {false, true}
@ATTRIBUTE toothed {false, true}
@ATTRIBUTE backbone {false, true}
@ATTRIBUTE breathes {false, true}
@ATTRIBUTE venomous {false, true}
@ATTRIBUTE fins {false, true}
@ATTRIBUTE legs INTEGER [0,9]
@ATTRIBUTE tail {false, true}
@ATTRIBUTE domestic {false, true}
@ATTRIBUTE catsize {false, true}
@ATTRIBUTE type { mammal, bird, reptile, fish, amphibian, insect, invertebrate }

@DATA
kangaroo,true,false,false,true,false,false,false,true,true,true,false,false,2,true,false,false,?
```
O que eu fiz foi criar uma nova instância, preenchendo-a com as características de um Canguru, exceto a classe (que queremos prever). Cangurus são mamíferos, então vamos ver como o sistema se comporta.

Vá para a aba Preprocess novamente, e carregue este arff. Após isso, vá para a aba Classify.

Clicando com o mouse direito na result list, aparece a opção "Load Model". Selecionando esta opção, escolha o arquivo .model que criamos antes. Agora, clique no botão More options...,  e em Output predictions selecione a opção PlainText.
Clique com o mouse direito na result list, novamente, e selecione a opção Re-evaluate model on current test set. E, VOILÁ!

![](weka8.png)

O que isso quer dizer é que, para a primeira instância lida do dataset, a classe era desconhecida, foi prevista como mamífero com porcentagem de erro 0 e porcentagem 100%.

Bem, foi isto, pessoal. Espero que tenham achado fácil e gostado de molhar a pontinha dos dedos no Weka e com Machine Learning, e até a próxima!