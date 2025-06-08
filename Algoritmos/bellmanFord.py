import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

manual = False

def bellman_ford_passo_a_passo(grafo, no_inicial):

    # grafo: O grafo a ser analisado, com arestas direcionadas e pesos.
    # no_inicial: O nó de onde todas as distâncias mais curtas serão calculadas.

    # --- 1. Inicialização ---
    distancias_minimas = {no: float('inf') for no in grafo.nodes}
    distancias_minimas[no_inicial] = 0
    
    no_predecessor = {no: None for no in grafo.nodes}

    posicoes_dos_nos = {
        'A': (0, 0), 'B': (1, 1), 'D': (1, -1),
        'C': (2, 0.8), 'E': (2, -0.8)
    }
    
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Função Auxiliar para Desenhar o Grafo ---
    def desenhar_estado_atual_do_grafo(passo_atual, predecessores, aresta_destacada=None):
        ax.clear()

        arestas_retas = [e for e in grafo.edges() if not grafo.has_edge(e[1], e[0])]
        arestas_em_arco = [e for e in grafo.edges() if grafo.has_edge(e[1], e[0])]

        caminho_anterior = set()
        if aresta_destacada:
            no_atual = aresta_destacada[0]
            while predecessores[no_atual] is not None:
                no_ant = predecessores[no_atual]
                caminho_anterior.add((no_ant, no_atual))
                no_atual = no_ant
        
        def obter_cor_aresta(aresta):
            if aresta_destacada and aresta == aresta_destacada:
                return 'red'
            elif aresta in caminho_anterior:
                return 'green'
            else:
                return 'black'

        cores_retas = [obter_cor_aresta(e) for e in arestas_retas]
        cores_arcos = [obter_cor_aresta(e) for e in arestas_em_arco]

        nx.draw_networkx_nodes(grafo, posicoes_dos_nos, ax=ax, node_color='lightblue', node_size=300)
        nx.draw_networkx_labels(grafo, posicoes_dos_nos, ax=ax, font_size=12, font_weight='bold')

        nx.draw_networkx_edges(grafo, posicoes_dos_nos, ax=ax, 
                               edgelist=arestas_retas, 
                               edge_color=cores_retas, 
                               width=3,
                               arrows=True,
                               arrowsize=20)

        nx.draw_networkx_edges(grafo, posicoes_dos_nos, ax=ax, 
                               edgelist=arestas_em_arco, 
                               edge_color=cores_arcos, 
                               width=3, 
                               connectionstyle='arc3,rad=0.2',
                               arrows=True,
                               arrowsize=20)
        
        todos_os_pesos = nx.get_edge_attributes(grafo, 'weight')
        rotulos_retos = {aresta: peso for aresta, peso in todos_os_pesos.items() if aresta in arestas_retas}
        
        nx.draw_networkx_edge_labels(grafo, posicoes_dos_nos, ax=ax, edge_labels=rotulos_retos, font_size=12)

        for u, v in arestas_em_arco:
            pos_u, pos_v = np.array(posicoes_dos_nos[u]), np.array(posicoes_dos_nos[v])
            pos_media = (pos_u + pos_v) / 2
            vetor_direcao = pos_v - pos_u
            vetor_perp = np.array([-vetor_direcao[1], vetor_direcao[0]])
            vetor_perp_norm = vetor_perp / np.linalg.norm(vetor_perp)
            deslocamento = 0.25
            pos_texto = pos_media + vetor_perp_norm * deslocamento
            peso = todos_os_pesos[(u, v)]
            ax.text(pos_texto[0], pos_texto[1], str(peso),
                    ha='center', va='center', fontsize=12,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.0))
        
        ax.set_title(f'Passo {passo_atual}: Relaxando Arestas\nDistâncias atuais: {distancias_minimas}')
        if(manual == False):
            plt.pause(3)
        else:
            a = input()

    # Desenha o estado inicial do grafo antes de começar os relaxamentos.
    desenhar_estado_atual_do_grafo(0, no_predecessor)

    # --- 2. Relaxamento das Arestas ---
    passo_atual = 1
    numero_de_nos = len(grafo.nodes)
    for _ in range(numero_de_nos - 1):
        for no_origem, no_destino, dados_aresta in grafo.edges(data=True):
            peso_aresta = dados_aresta['weight']

            if distancias_minimas[no_origem] != float('inf') and distancias_minimas[no_origem] + peso_aresta < distancias_minimas[no_destino]:
                # distancias_minimas_anteriores = distancias_minimas.copy()
                no_predecessor_anterior = no_predecessor.copy()
                
                distancias_minimas[no_destino] = distancias_minimas[no_origem] + peso_aresta
                no_predecessor[no_destino] = no_origem
                
                desenhar_estado_atual_do_grafo(passo_atual, no_predecessor_anterior, aresta_destacada=(no_origem, no_destino))
                passo_atual += 1

    # --- 3. Verificação de Ciclos de Peso Negativo ---
    for no_origem, no_destino, dados_aresta in grafo.edges(data=True):
        peso_aresta = dados_aresta['weight']
        if distancias_minimas[no_origem] != float('inf') and distancias_minimas[no_origem] + peso_aresta < distancias_minimas[no_destino]:
            print("--------------------------------------------------")
            print("AVISO: O grafo contém um ciclo de peso negativo!")
            print("As distâncias mínimas não podem ser determinadas.")
            print("--------------------------------------------------")
            plt.show()
            return None, None

    ax.set_title("Execução Finalizada - Distâncias Mínimas Encontradas")
    plt.show()
    return distancias_minimas, no_predecessor



# Criação de um grafo direcionado e ponderado de exemplo
grafo = nx.DiGraph()
grafo.add_weighted_edges_from([
    ('A', 'B', 4), ('A', 'D', 2), ('B', 'C', 2),
    ('B', 'E', 3), ('B', 'D', 3), ('D', 'B', 1),
    ('D', 'C', 4), ('D', 'E', 5), ('E', 'C', -5)
])

# 4. Rodando o algoritmo
plt.ion() 
distancias_finais, predecessores_finais = bellman_ford_passo_a_passo(grafo, 'A')
plt.ioff()

# Apresentando os Resultados Finais
if distancias_finais is not None:
    print("\n--- Distâncias mínimas finais a partir da origem 'A': ---")
    for no, valor_distancia in distancias_finais.items():
        print(f"Nó {no}: {valor_distancia}")
