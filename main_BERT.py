import toyplot
import torch
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification
import numpy as np
from torch_geometric.data import Data
from torch.utils.data import ConcatDataset





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

                        Caricamento tokenizzatore e modello BERT

"""





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Carica il modello BERT e il tokenizer
BERT_model = BertForSequenceClassification.from_pretrained("./bert_model")
model_name = "zhihan1996/DNA_bert_6"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Sposta il modello sulla GPU (se disponibile)
BERT_model.to(device)





"""

                        Creazione del dataset per il modello

"""





def get_feature_matrix(nodes):
    BERT_model.eval()

    features_matrix = []
    for node in nodes:
        # Sposta gli input sulla GPU
        inputs = tokenizer(node, return_tensors="pt", max_length=128, padding='max_length', truncation=True).to(device)

        # Estrai embedding BERT
        with torch.no_grad():  # Non serve il calcolo del gradiente
            outputs = BERT_model(**inputs, output_hidden_states=True)

        # Ottieni gli embedding dal livello nascosto (hidden states)
        hidden_states = outputs.hidden_states
        last_hidden_state = hidden_states[-1]
        embedding = last_hidden_state.mean(dim=1)  # Puoi usare mean pooling sull'intera sequenza
        # Sposta il risultato sulla CPU per appendere
        features_matrix.append(embedding.squeeze(0).cpu().numpy())

    return features_matrix


def create_graph_data(sequences, chimeric):
    data_list = []
    k = 0
    for seq in sequences:
        kmers = get_kmer(seq, k=6)  # ottengo i kmer della sequenza
        edges = get_debruijn_edges(kmers)  # creo gli archi del grafo di De Bruijn
        nodes, adjacency_matrix = create_adjacency_matrix(edges)  # creo la matrice di adiacenza del grafo
        features_matrix = get_feature_matrix(nodes)  # creo la matrice delle feature dei nodi
        features_matrix = np.array(features_matrix)

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
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

        # Sposta il tensore delle feature su GPU
        x = torch.tensor(features_matrix, dtype=torch.float).to(device)

        # Crea un tensore di PyTorch con le etichette per ogni grafo
        y = torch.tensor([1 if chimeric else 0], dtype=torch.long).to(device)

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

dataset = ConcatDataset([chimeric_data_list, not_chimeric_data_list])


torch.save(chimeric_data_list, 'chimeric_dataset_BERT.pt')
torch.save(not_chimeric_data_list, 'not_chimeric_dataset_BERT.pt')
