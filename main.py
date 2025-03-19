import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import cv2
import imageio
from PIL import Image, ImageDraw
import os

import heapq  # Para a prioridade do A*
import time   # Para medir desempenho

class Labirinto:
    def __init__(self, matriz=None, caminho_imagem=None):
        if caminho_imagem:
            self.carregar_imagem(caminho_imagem)
        elif matriz is not None:
            self.matriz = np.array(matriz)
        else:
            raise ValueError("Deve fornecer uma matriz ou caminho de imagem")

        self.altura = self.matriz.shape[0]
        self.largura = self.matriz.shape[1]
        self.inicio = None
        self.objetivo = None

        self.encontrar_pontos()

    def carregar_imagem(self, caminho_imagem):
        """Carrega e processa uma imagem de labirinto"""
        img = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)
        _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        self.matriz = (img_bin / 255).astype(int)
        if np.mean(self.matriz) > 0.5:
            self.matriz = 1 - self.matriz

    def encontrar_pontos(self):
        """Encontra início e objetivo na matriz (assumindo 2 e 3)"""
        for i in range(self.altura):
            for j in range(self.largura):
                if self.matriz[i][j] == 2:
                    self.inicio = (i, j)
                elif self.matriz[i][j] == 3:
                    self.objetivo = (i, j)

    def eh_valido(self, x, y):
        """Verifica se a posição (x,y) está dentro dos limites e não é parede."""
        return 0 <= x < self.altura and 0 <= y < self.largura and self.matriz[x][y] != 1

    def obter_vizinhos(self, x, y):
        """Retorna posições vizinhas válidas (cima, direita, baixo, esquerda)."""
        movimentos = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        vizinhos = []
        for dx, dy in movimentos:
            nx, ny = x + dx, y + dy
            if self.eh_valido(nx, ny):
                vizinhos.append((nx, ny))
        return vizinhos

    def busca_largura(self, passo_a_passo=False, delay=0.3):
        """Implementa a busca em largura com geração de imagens do progresso. BFS."""
        if not self.inicio or not self.objetivo:
            raise ValueError("Pontos de início ou objetivo não definidos no labirinto.")
        
        fila = deque([self.inicio])
        visitados = {self.inicio: None}
        nos_expandidos = 0
        image_paths = []

        while fila:
            atual = fila.popleft()
            nos_expandidos += 1

            if atual == self.objetivo:
                caminho = []
                while atual:
                    caminho.append(atual)
                    atual = visitados[atual]
                caminho.reverse()

                for i, passo in enumerate(caminho):
                    output_path = f"temp_path_{i}.png"
                    self.create_labyrinth_image(caminho[:i + 1], output_path)
                    image_paths.append(output_path)

                return caminho, nos_expandidos, image_paths

            for vizinho in self.obter_vizinhos(*atual):
                if vizinho not in visitados:
                    fila.append(vizinho)
                    visitados[vizinho] = atual
    
    def busca_profundidade(self, passo_a_passo=False):
        """Implementa a Busca em Profundidade (DFS) com contagem de nós expandidos e geração de GIF."""
        if not self.inicio or not self.objetivo:
            raise ValueError("Pontos de início ou objetivo não definidos no labirinto.")

        pilha = [self.inicio]
        visitados = {self.inicio: None}
        nos_expandidos = 0  # Contador de nós expandidos
        image_paths = []

        while pilha:
            atual = pilha.pop()
            nos_expandidos += 1  # Incrementa o contador ao expandir um nó

            # Salva imagens do progresso
            if passo_a_passo:
                caminho_atual = []
                temp = atual
                while temp:
                    caminho_atual.append(temp)
                    temp = visitados.get(temp)
                caminho_atual.reverse()
                output_path = f"dfs_temp_{len(image_paths)}.png"
                self.create_labyrinth_image(caminho_atual, output_path)
                image_paths.append(output_path)

            # Se encontrou o objetivo, reconstrói o caminho
            if atual == self.objetivo:
                caminho = []
                while atual:
                    caminho.append(atual)
                    atual = visitados[atual]
                return caminho[::-1], nos_expandidos, image_paths

            # Explora os vizinhos
            for vizinho in self.obter_vizinhos(*atual):
                if vizinho not in visitados:
                    pilha.append(vizinho)
                    visitados[vizinho] = atual

        raise ValueError("Caminho não encontrado com DFS.")

    
    def busca_a_estrela(self, passo_a_passo=False):
        """Implementa o algoritmo A* com contagem de nós expandidos e geração de GIF."""
        if not self.inicio or not self.objetivo:
            raise ValueError("Pontos de início ou objetivo não definidos no labirinto.")

        def heuristica(a, b):
            """Calcula a heurística Manhattan."""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        fila_prioridade = []
        heapq.heappush(fila_prioridade, (0, self.inicio))
        custo_atual = {self.inicio: 0}
        visitados = {self.inicio: None}
        nos_expandidos = 0  # Contador de nós expandidos
        image_paths = []

        while fila_prioridade:
            _, atual = heapq.heappop(fila_prioridade)
            nos_expandidos += 1  # Incrementa o contador de nós expandidos

            # Salva imagens do progresso
            if passo_a_passo:
                caminho_atual = []
                temp = atual
                while temp:
                    caminho_atual.append(temp)
                    temp = visitados.get(temp)
                caminho_atual.reverse()
                output_path = f"a_star_temp_{len(image_paths)}.png"
                self.create_labyrinth_image(caminho_atual, output_path)
                image_paths.append(output_path)

            # Reconstrói o caminho se o objetivo foi alcançado
            if atual == self.objetivo:
                caminho = []
                while atual:
                    caminho.append(atual)
                    atual = visitados[atual]
                return caminho[::-1], nos_expandidos, image_paths

            # Explora os vizinhos
            for vizinho in self.obter_vizinhos(*atual):
                novo_custo = custo_atual[atual] + 1
                if vizinho not in custo_atual or novo_custo < custo_atual[vizinho]:
                    custo_atual[vizinho] = novo_custo
                    prioridade = novo_custo + heuristica(vizinho, self.objetivo)
                    heapq.heappush(fila_prioridade, (prioridade, vizinho))
                    visitados[vizinho] = atual

        raise ValueError("Caminho não encontrado com A*.")
    
    def comparar_buscas(self):
        """Compara BFS, DFS e A* e exibe o desempenho, incluindo nós expandidos."""
        resultados = {}

        for metodo in ["busca_largura", "busca_profundidade", "busca_a_estrela"]:
            inicio_tempo = time.time()
            resultado = getattr(self, metodo)(passo_a_passo=False)
            tempo_exec = time.time() - inicio_tempo

            # Extrai os dados retornados pelo método de busca
            caminho = resultado[0]  # Primeiro elemento: o caminho
            nos_expandidos = resultado[1] if len(resultado) > 1 else None  # Segundo elemento: nós expandidos
            passos = len(caminho) - 1 if caminho else None  # Calcula o número de passos no caminho

            resultados[metodo] = {
                "caminho": caminho,
                "tempo_execucao": tempo_exec,
                "passos": passos,
                "nos_expandidos": nos_expandidos
            }

        # Exibindo os resultados formatados
        print("Resultados da comparação:")
        for metodo, dados in resultados.items():
            print(f"{metodo}:")
            print(f"  Tempo de execução: {dados['tempo_execucao']:.6f} segundos")
            print(f"  Passos no caminho: {dados['passos']}")
            print(f"  Nós expandidos: {dados['nos_expandidos']}\n")

        return resultados

    def create_labyrinth_image(self, path, output_path):
        """Gera uma imagem visualizando o labirinto e o caminho."""
        tile_size = 20
        img_width = self.largura * tile_size
        img_height = self.altura * tile_size

        img = Image.new("RGB", (img_width, img_height), "white")
        draw = ImageDraw.Draw(img)

        # Desenhando o labirinto
        for y in range(self.altura):
            for x in range(self.largura):
                cor = "white"
                if self.matriz[y][x] == 1:  # Paredes
                    cor = "black"
                elif self.matriz[y][x] == 2:  # Início
                    cor = "blue"
                elif self.matriz[y][x] == 3:  # Objetivo
                    cor = "red"
                draw.rectangle([x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size], fill=cor)

        # Desenhando o caminho
        for (y, x) in path:  # Note que as coordenadas estão invertidas para o desenho
            draw.rectangle([x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size], fill="green")

        img.save(output_path)

    def create_gif(self, image_paths, gif_output_path):
        """Gera um GIF animado com base nas imagens do progresso."""
        if not image_paths:
            raise ValueError("Lista de caminhos para imagens está vazia!")

        images = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                images.append(Image.open(image_path))
            else:
                print(f"Imagem não encontrada: {image_path}")
        
        if not images:
            raise ValueError("Nenhuma imagem válida encontrada para criar o GIF.")

        imageio.mimsave(gif_output_path, images, duration=0.5)

    # Limpar arquivos temporários
    def limpar_arquivos_temporarios(self, image_paths):
        """Remove arquivos temporários de forma segura."""
        import os
        for image_path in image_paths:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
                    #print(f"Arquivo excluído: {image_path}")
                else:
                    print(f"Arquivo já não existe: {image_path}")
            except Exception as e:
                print(f"Erro ao excluir {image_path}: {e}")

labirinto = Labirinto(matriz = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0],  # 2 = S (start), 3 = G (goal)
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 3],  # 3 = Goal (G)
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
])

# BFS
caminho_bfs, nos_expandidos_bfs, image_paths_bfs = labirinto.busca_largura(passo_a_passo=True)
labirinto.create_gif(image_paths_bfs, "bfs_path.gif")  # Cria o GIF antes da exclusão

# DFS
caminho_dfs, nos_expandidos_dfs, image_paths_dfs = labirinto.busca_profundidade(passo_a_passo=True)
labirinto.create_gif(image_paths_dfs, "dfs_path.gif")  # Cria o GIF antes da exclusão

# A*
caminho_a_star, nos_expandidos_a_star, image_paths_a_star = labirinto.busca_a_estrela(passo_a_passo=True)
labirinto.create_gif(image_paths_a_star, "a_star_path.gif")  # Cria o GIF antes da exclusão

# Limpar arquivos temporários após criar os GIFs
labirinto.limpar_arquivos_temporarios(image_paths_dfs + image_paths_a_star)

resultados = labirinto.comparar_buscas()

labirinto.limpar_arquivos_temporarios(image_paths_bfs)