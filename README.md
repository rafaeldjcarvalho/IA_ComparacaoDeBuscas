# Simulador de Estratégias de Busca em Labirintos

## Descrição
Este projeto implementa um simulador de busca em labirintos utilizando diferentes estratégias de busca. O sistema lê um labirinto a partir de uma matriz ou de uma imagem e encontra o caminho mais eficiente entre um ponto de início e um destino. O projeto também gera uma animação da busca realizada, permitindo visualizar o percurso.

## Funcionalidades
- Carregamento de labirinto a partir de uma matriz ou imagem.
- Algoritmos implementados:
  - **Busca em Largura (BFS)**
  - **Busca em Profundidade (DFS)**
  - **A* (A estrela)**
  - **Busca baseada em IA (rede neural convolucional)**
- Geração de imagens do percurso e criação de GIF animado.
- Análise do tempo de execução e quantidade de nós expandidos por algoritmo.

## Algoritmos de Busca Implementados
### 1. Busca em Largura (BFS)
A Busca em Largura explora o labirinto nível por nível, garantindo sempre o caminho mais curto possível. Utiliza uma fila (FIFO) para armazenar os nós a serem explorados.

- **Vantagens:** Garante o caminho mais curto.
- **Desvantagens:** Alto consumo de memória em labirintos grandes.

### 2. Busca em Profundidade (DFS)
A Busca em Profundidade segue um caminho até encontrar um beco sem saída e depois retrocede para explorar novas possibilidades. Utiliza uma pilha (LIFO) para armazenar os nós.

- **Vantagens:** Menor uso de memória comparado ao BFS.
- **Desvantagens:** Pode entrar em loops ou seguir caminhos muito longos sem encontrar a solução ideal.

### 3. Algoritmo A* (A estrela)
O algoritmo A* usa uma heurística para priorizar caminhos mais promissores. Ele combina a distância percorrida com uma estimativa da distância restante.

- **Vantagens:** Altamente eficiente, geralmente encontra o caminho mais curto rapidamente.
- **Desvantagens:** Dependente da função heurística escolhida.

### 4. Busca baseada em IA
A estratégia baseada em inteligência artificial utiliza uma rede neural convolucional para aprender padrões de caminhos e prever a melhor rota. É um método experimental que pode ser ajustado conforme o treinamento da IA.

- **Vantagens:** Pode ser altamente eficiente se bem treinado.
- **Desvantagens:** Requer treinamento prévio e pode apresentar resultados inconsistentes sem um conjunto de dados adequado.

## Dependências
Para executar o projeto, instale as bibliotecas necessárias com:

```sh
pip install numpy matplotlib opencv-python imageio pillow
```

## Execução
Para rodar o simulador, basta executar o script principal:

```sh
python main.py
```

O sistema carregará um labirinto e executará os algoritmos de busca, gerando um GIF com a animação do percurso encontrado.

## Exemplo de Labirinto
Representação de um labirinto onde:
- `0` = caminho livre
- `1` = parede
- `2` = ponto de início
- `3` = destino

```python
matriz = [
    [2, 0, 1, 0, 0, 0],
    [1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 3, 0]
]
```

## Licença
Este projeto é distribuído sob a licença MIT. Sinta-se livre para modificá-lo e utilizá-lo como desejar.


