import toyplot
import torch
import numpy as np
from torch_geometric.data import Data





"""

                    Creazione grafo di De Bruijn

"""





def estrai_sequenze_geni(file_fastq):
    sequenze_geni = []
    k = 0
    with open(file_fastq, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            sequence = lines[i + 1].strip()
            if len(sequence) > 6:
                sequenze_geni.append(sequence)
                k += 1
            if (k == 960):
                break
    return sequenze_geni


def get_kmer(sequence, k=4):
    kmers = []
    for i in range(0, len(sequence)):
        if len(sequence[i:i + k]) != k:
            continue
        kmers.append(sequence[i:i + k])
    return kmers


def get_debruijn_edges(kmers):
    edges = set()
    for k1 in kmers:
        for k2 in kmers:
            if k1 != k2:
                if k1[1:] == k2[:-1]:
                    edges.add((k1, k2))
                if k1[:-1] == k2[1:]:
                    edges.add((k2, k1))
    return edges


def plot_debruijn_graph(edges, width=2000, height=2000):
    graph = toyplot.graph(
        [i[0] for i in edges],
        [i[1] for i in edges],
        width=width,
        height=height,
        tmarker=">",
        vsize=25,
        vstyle={"stroke": "black", "stroke-width": 2, "fill": "none"},
        vlstyle={"font-size": "11px"},
        estyle={"stroke": "black", "stroke-width": 2},
        layout=toyplot.layout.FruchtermanReingold(edges=toyplot.layout.CurvedEdges())
    )
    return graph


def create_adjacency_matrix(edges):
    nodes = sorted(set([e[0] for e in edges] + [e[1] for e in edges]))
    node_index = {node: i for i, node in enumerate(nodes)}

    size = len(nodes)
    adjacency_matrix = np.zeros((size, size))

    for edge in edges:
        from_node, to_node = edge
        i = node_index[from_node]
        j = node_index[to_node]
        adjacency_matrix[i, j] = 1

    return nodes, adjacency_matrix


def print_adjacency_matrix(nodes, adjacency_matrix):
    print("Adjacency Matrix:")
    print("   " + " ".join(f"{node:>3}" for node in nodes))

    for i, row in enumerate(adjacency_matrix):
        print(f"{nodes[i]:>3} " + " ".join(f"{cell:>3}" for cell in row))





"""

                    One-Hot Encoding

"""





onehot = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'N': [0, 0, 0, 0],  # Per nucleotidi sconosciuti o gap
}


def get_feature_matrix(nodes):
    features_matrix = []

    # Itera attraverso i nodi, ciascun nodo può essere un k-mer (sequenza) o un singolo nucleotide
    for node in nodes:
        # Se il nodo è un k-mer, converti ciascun nucleotide in one-hot encoding e concatenali
        onehot_representation = []
        for nucleotide in node:
            onehot_representation.append(onehot.get(nucleotide, onehot['N']))

        # Aggiungi la rappresentazione one-hot completa del nodo alla matrice delle feature
        features_matrix.append(onehot_representation)
    # Converte la lista in una matrice numpy
    return np.array(features_matrix)





"""

                    Creazione del dataset per il modello

"""





def create_graph_data(sequences, chimeric):
    data_list = []
    k = 0
    for seq in sequences:
        kmers = get_kmer(seq, k = 6)  #ottengo i kmer della sequenza
        edges = get_debruijn_edges(kmers)  #creo gli archi del grafo di De Bruijn
        nodes, adjacency_matrix = create_adjacency_matrix(edges)  #creo la matrice di adiacenza del grafo
        features_matrix = get_feature_matrix(nodes)  #creo la matrice delle feature dei nodi, che continene per ogni riga l'encoding BERT del nodo

        # Crea una lista vuota per memorizzare gli indici degli archi
        edge_index = []

        # Crea un dizionario che mappa ogni nodo al suo indice numerico
        node_index = {node: i for i, node in enumerate(nodes)}

        # Itera su ciascun arco nel grafo
        for edge in edges:
            # Ottieni gli indici dei nodi connessi dall'arco
            i, j = node_index[edge[0]], node_index[edge[1]]
            # Aggiungi una coppia di indici all'elenco degli indici degli archi
            edge_index.append([i, j])

        # Converti l'elenco degli indici degli archi in un tensore di PyTorch
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        node_features = []
        for onehot_list in features_matrix:
            # Ogni k-mer ha una lista di one-hot separati (uno per nucleotide)
            # Li appiattiamo lungo la prima dimensione (concatenazione lungo l'asse orizzontale)
            flattened = torch.cat([torch.tensor(vec, dtype=torch.float) for vec in onehot_list], dim=0)
            node_features.append(flattened)

        # Convertiamo la lista di feature dei nodi in un tensore
        x = torch.stack(node_features)

        # Converte la matrice delle feature in un tensore di PyTorch
        #x = torch.tensor(features_matrix, dtype=torch.float)

        # Crea un tensore di PyTorch con le etichette per ogni nodo
        y = torch.tensor([1 if chimeric else 0], dtype=torch.long)

        # Crea un oggetto Data che rappresenta un grafo
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

        k += 1
        print(f'{k:4d} Graph created: ')
        print(data)
    return data_list





"""

                            MAIN

"""





chimeric = 'data/100_genes_Datasets_input/dataset_chimeric2.fastq'
not_chimeric = 'data/100_genes_Datasets_input/dataset_non_chimeric.fastq'
chimeric_sequences = estrai_sequenze_geni(chimeric)
not_chimeric_sequences = estrai_sequenze_geni(not_chimeric)
chimeric_data_list = create_graph_data(chimeric_sequences, chimeric=True)
not_chimeric_data_list = create_graph_data(not_chimeric_sequences, chimeric=False)


torch.save(chimeric_data_list, 'dataset/chimeric_dataset_ONE-HOT.pt')
torch.save(not_chimeric_data_list, 'dataset/not_chimeric_dataset_ONE-HOT.pt')