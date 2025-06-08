import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd 

def floyd_warshall_passo_a_passo(grafo):
    
    # --- 1. Inicialização ---
    nos = list(grafo.nodes)
    num_nos = len(nos)
    
    # Mapeia os nomes dos nós (ex: 'A', 'B') para índices (0, 1) para facilitar o acesso à matriz.
    mapa_indices = {no: i for i, no in enumerate(nos)}
    mapa_nomes = {i: no for i, no in enumerate(nos)}

    # `dist`: Matriz que guarda a menor distância conhecida entre cada par de nós.
    # Inicialmente, todas são 'infinito', exceto as distâncias para si mesmo (0)
    # e as arestas diretas existentes.
    dist = [[float('inf')] * num_nos for _ in range(num_nos)]
    
    # `prox_no`: Matriz que armazena o 'próximo' nó no caminho mais curto de i para j.
    # Essencial para reconstruir o caminho depois.
    prox_no = [[None] * num_nos for _ in range(num_nos)]

    for i in range(num_nos):
        dist[i][i] = 0
        prox_no[i][i] = i

    for u, v, dados in grafo.edges(data=True):
        i, j = mapa_indices[u], mapa_indices[v]
        dist[i][j] = dados['weight']
        prox_no[i][j] = j # Para ir de u a v, o próximo nó é v

    # `posicoes_dos_nos`: Define as posições fixas para a visualização.
    posicoes_dos_nos = nx.spring_layout(grafo, k=5, iterations=100, seed=13648)
    
    # --- Função Auxiliar para Imprimir a Matriz de Distâncias ---
    def imprimir_matriz_distancias(matriz_dist, titulo="Matriz de Distâncias"):
        # Converte a matriz para um DataFrame do Pandas para uma exibição clara
        df = pd.DataFrame(matriz_dist, index=nos, columns=nos)
        print(f"\n--- {titulo} ---")
        print(df.round(2)) # Arredonda para 2 casas decimais

    # --- Função Auxiliar para Desenhar o Grafo ---
    def desenhar_estado_atual_do_grafo(k_atual, caminho_destacado=None):
        
        plt.clf()

        # Define cores dos nós
        cores_nos = ['lightblue'] * num_nos
        if caminho_destacado:
            i, k, j = caminho_destacado
            cores_nos[i] = 'yellow'      # Nó de origem
            cores_nos[k] = 'green' # Nó intermediário
            cores_nos[j] = 'red'    # Nó de destino

        # Define cores das arestas
        cores_arestas = ['gray'] * len(grafo.edges)
        if caminho_destacado:
            i, k, j = caminho_destacado
            nome_i, nome_k, nome_j = mapa_nomes[i], mapa_nomes[k], mapa_nomes[j]
            arestas = list(grafo.edges)
            # Tenta destacar as arestas (i, k) e (k, j) se elas existirem
            if (nome_i, nome_k) in arestas:
                cores_arestas[arestas.index((nome_i, nome_k))] = 'green'
            if (nome_k, nome_j) in arestas:
                cores_arestas[arestas.index((nome_k, nome_j))] = 'green'

        # Desenha o grafo
        rotulos_pesos_arestas = nx.get_edge_attributes(grafo, 'weight')
        nx.draw(grafo, posicoes_dos_nos, with_labels=True, node_color=cores_nos,
                edge_color=cores_arestas, node_size=2000, font_size=10, font_weight='bold', width=2)
        nx.draw_networkx_edge_labels(grafo, posicoes_dos_nos, edge_labels=rotulos_pesos_arestas)

        titulo = f"Iteração com k = '{mapa_nomes[k_atual]}' (Nó Intermediário)"
        if caminho_destacado:
             nome_i, nome_k, nome_j = mapa_nomes[i], mapa_nomes[k], mapa_nomes[j]
             titulo += f"\nAtualização via: {nome_i} -> {nome_k} -> {nome_j}"

        plt.title(titulo)
        plt.pause(2.5) # Pausa para visualização

    # Mostra a matriz inicial
    imprimir_matriz_distancias(dist, "Matriz de Distâncias Inicial (D^0)")
    
    # --- 2. Iterações Principais ---
    # O laço 'k' é o mais externo e define o nó intermediário que será considerado.
    for k in range(num_nos):
        print(f"\n\n===== CONSIDERANDO NÓ INTERMEDIÁRIO k = '{mapa_nomes[k]}' =====")
        desenhar_estado_atual_do_grafo(k) # Mostra qual 'k' estamos usando
        
        for i in range(num_nos):
            for j in range(num_nos):
                # Condição de Atualização de Floyd-Warshall:
                # O caminho de 'i' para 'j' passando por 'k' é mais curto?
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist_anterior = dist[i][j]
                    dist[i][j] = dist[i][k] + dist[k][j]
                    prox_no[i][j] = prox_no[i][k] # O próximo nó de i para j é o mesmo que de i para k
                    
                    # Log e visualização da atualização
                    print(f"Atualização! Dist({mapa_nomes[i]},{mapa_nomes[j]}): {dist_anterior} -> {dist[i][j]:.2f} "
                          f"(via {mapa_nomes[i]}->{mapa_nomes[k]}->{mapa_nomes[j]})")
                    desenhar_estado_atual_do_grafo(k, caminho_destacado=(i, k, j))

        # Mostra a matriz de distâncias ao final de cada iteração de 'k'
        imprimir_matriz_distancias(dist, f"Matriz Após Iteração k='{mapa_nomes[k]}' (D^{k+1})")

    # --- 3. Verificação de Ciclos de Peso Negativo ---
    # Após o algoritmo, se a distância de um nó para ele mesmo for negativa,
    # há um ciclo de peso negativo no grafo.
    for i in range(num_nos):
        if dist[i][i] < 0:
            print("\n--------------------------------------------------")
            print("AVISO: O grafo contém um ciclo de peso negativo!")
            print("As distâncias mínimas não são confiáveis.")
            print("--------------------------------------------------")
            plt.show()
            return None, None
    
    plt.show()
    
    # Converte as matrizes finais de volta para usar nomes de nós em vez de índices
    dist_final = {u: {v: dist[mapa_indices[u]][mapa_indices[v]] for v in nos} for u in nos}
    prox_no_final = {u: {v: mapa_nomes.get(prox_no[mapa_indices[u]][mapa_indices[v]]) for v in nos} for u in nos}

    return dist_final, prox_no_final

def reconstruir_caminho(prox_no, no_origem, no_destino):
    """Reconstrói o caminho mais curto usando a matriz de próximos nós."""
    if prox_no[no_origem][no_destino] is None:
        return "Nenhum caminho encontrado"
    
    caminho = [no_origem]
    u = no_origem
    while u != no_destino:
        # Pega o próximo nó no caminho de u para o destino
        # E o converte de volta para o nome do nó (ex: 'A')
        u = prox_no[u][no_destino]
        if u is None: # Se não houver próximo nó, o caminho está quebrado (não deveria acontecer)
            return "Erro na reconstrução"
        caminho.append(u)
    return " -> ".join(caminho)

# --- Criação do Grafo de Exemplo ---
grafo = nx.DiGraph()
grafo.add_weighted_edges_from([
    ('A', 'B', 4), ('A', 'D', 2), ('B', 'C', 2),
    ('B', 'E', 3), ('B', 'D', 3), ('D', 'B', 1),
    ('D', 'C', 4), ('D', 'E', 5), ('E', 'C', -5)
])

# --- 4. Rodando o algoritmo Floyd-Warshall com visualização ---
plt.ion()
distancias_finais, proximos_nos_finais = floyd_warshall_passo_a_passo(grafo)
plt.ioff()

# --- Apresentando os Resultados Finais ---
if distancias_finais is not None:
    print("\n\n################ RESULTADOS FINAIS ################")
    print("\n--- Matriz de Distâncias Mínimas Finais ---")
    df_dist = pd.DataFrame(distancias_finais).T 
    print(df_dist)

    print("\n--- Matriz de Próximos Nós (para reconstrução de caminho) ---")
    df_prox = pd.DataFrame(proximos_nos_finais).T
    print(df_prox)

    print("\n--- Exemplo de Reconstrução de Caminho ---")
    origem = 'A'
    destino = 'C'
    caminho = reconstruir_caminho(proximos_nos_finais, origem, destino)
    distancia = distancias_finais[origem][destino]
    print(f"Caminho mais curto de {origem} para {destino}: {caminho} (Custo: {distancia})")
    
    origem = 'A'
    destino = 'E'
    caminho = reconstruir_caminho(proximos_nos_finais, origem, destino)
    distancia = distancias_finais[origem][destino]
    print(f"Caminho mais curto de {origem} para {destino}: {caminho} (Custo: {distancia})")