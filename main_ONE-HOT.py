import toyplot
from sklearn.metrics import roc_curve, auc
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold





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
        features_matrix = get_feature_matrix(
            nodes)  #creo la matrice delle feature dei nodi, che continene per ogni riga l'encoding BERT del nodo

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

                        Modello GNN

"""




in_channels = 24
hidden_channels = 128
out_channels = 2


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, hidden_channels)
        self.conv6 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Passaggio attraverso i livelli GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv6(x, edge_index)

        # Pooling globale (media) per ottenere un vettore per ogni grafo
        x = global_mean_pool(x, batch)

        # Classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)





"""

                            Addestramento e validazione

"""





def train(model, train_loader, val_loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    epochs = 200  # Numero di epoche per cui allenare il modello

    model.train()  # Imposta il modello in modalità di addestramento

    # Ciclo principale di addestramento per ciascuna epoca
    for epoch in range(epochs + 1):
        # Inizializza le variabili per tenere traccia della perdita totale e dell'accuratezza
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Addestramento del modello su mini-batch
        for data in train_loader:
            optimizer.zero_grad()  # Resetto i gradienti dell'ottimizzatore
            out = model(data)  # Passo i dati attraverso il modello
            loss = criterion(out, data.y)  # Passo i dati attraverso il modello

            # Aggiorno la perdita totale e l'accuratezza per questa epoca
            total_loss += loss / len(train_loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(train_loader)

            loss.backward()  # Calcolo i gradienti per la backpropagation
            optimizer.step()  # Aggiorno i parametri del modello usando l'ottimizzatore

            val_loss, val_acc = validate(model, val_loader)  # Validazione del modello sul set di validazione

        # Stampo le metriche ogni 10 epoche per monitorare l'addestramento
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                  f'| Train Acc: {acc * 100:>5.2f}% '
                  f'| Val Loss: {val_loss:.2f} '
                  f'| Val Acc: {val_acc * 100:.2f}%')

    return model


@torch.no_grad()
def validate(model, loader):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    loss = 0
    acc = 0

    for data in loader:
        out = model(data)

        # Calcola la perdita per il batch corrente
        loss += criterion(out, data.y) / len(loader)

        # Calcola l'accuratezza
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

    return loss, acc


@torch.no_grad()
def test(model, loader):
    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    all_preds = []
    all_labels = []
    loss = 0
    acc = 0

    for data in loader:
        out = model(data)

        # Calcola la perdita per il batch corrente
        loss += criterion(out, data.y) / len(loader)

        # Calcola l'accuratezza
        acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

        # Estrai le probabilità predette per la classe positiva
        prob = torch.softmax(out, dim=1)[:, 1]  # Probabilità della classe positiva (1)

        # Salva le probabilità e le etichette vere
        all_preds.append(prob.cpu().numpy())  # Porta i dati su CPU per sklearn
        all_labels.append(data.y.cpu().numpy())

    # Concatena i risultati di tutti i batch
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calcola i valori della curva ROC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc_value = auc(fpr, tpr)

    # Disegna la curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_value:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    return loss, acc


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()  # Confronta le previsioni con le etichette reali e calcola la  percentuale di corrispondenza


"""

                            MAIN

"""



chimeric = 'data/100_genes_Datasets_input/dataset_chimeric2.fastq'
not_chimeric = 'data/100_genes_Datasets_input/dataset_non_chimeric.fastq'
chimeric_sequences = estrai_sequenze_geni(chimeric)
not_chimeric_sequences = estrai_sequenze_geni(not_chimeric)
chimeric_data_list = create_graph_data(chimeric_sequences, chimeric=True)
not_chimeric_data_list = create_graph_data(not_chimeric_sequences, chimeric=False)





"""

                            K-Fold Validation

"""





dataset = ConcatDataset([chimeric_data_list, not_chimeric_data_list])
test_size = int(0.1 * len(dataset))
train_val_size = len(dataset) - test_size

train_val_set, test_set = torch.utils.data.random_split(dataset, [train_val_size, test_size])

test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
kf = KFold(n_splits=5, shuffle=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_set)):
    train_subset = Subset(train_val_set, train_idx)
    val_subset = Subset(train_val_set, val_idx)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=True)

    model = GCN(in_channels, hidden_channels, out_channels)

    trained_model = train(model, train_loader, val_loader)
    test_loss, test_acc = test(trained_model, test_loader)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%')
