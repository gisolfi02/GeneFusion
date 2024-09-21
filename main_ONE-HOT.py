import toyplot
from sklearn.metrics import roc_curve, auc
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import seaborn as sns




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

                        Modello GNN

"""





in_channels = 24
hidden_channels = 16
out_channels = 1

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Passaggio attraverso i livelli GCN
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # Pooling globale (media) per ottenere un vettore per ogni grafo
        x = global_mean_pool(x, batch)

        # Classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        # Applica la funzione sigmoid per il ritorno
        return torch.sigmoid(x)





"""

                            Addestramento e validazione

"""





def train(model, loader):
    criterion = torch.nn.BCELoss()  # Usare BCELoss con sigmoid nel modello
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    epochs = 200  # Numero di epoche per cui allenare il modello

    model.train()  # Imposta il modello in modalità di addestramento

    # Ciclo principale di addestramento per ciascuna epoca
    for epoch in range(epochs + 1):
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Addestramento del modello su mini-batch
        for data in loader:
            optimizer.zero_grad()  # Resetto i gradienti dell'ottimizzatore
            out = model(data)  # Passo i dati attraverso il modello
            loss = criterion(out, data.y.unsqueeze(1).float())  # Calcolo della perdita

            # Aggiorno la perdita totale e l'accuratezza per questa epoca
            total_loss += loss / len(loader)
            predictions = (out > 0.5).float()  # Converti le probabilità in etichette binarie
            acc += accuracy(predictions, data.y) / len(loader)

            loss.backward()  # Calcolo i gradienti per la backpropagation
            optimizer.step()  # Aggiorno i parametri del modello usando l'ottimizzatore

            val_loss, val_acc = validate(model, val_loader)  # Validazione del modello sul set di validazione

        # Stampo le metriche ogni 10 epoche per monitorare l'addestramento
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                  f'| Train Acc: {acc:>5.2f}% '
                  f'| Val Loss: {val_loss:.2f} '
                  f'| Val Acc: {val_acc:.2f}%')

    return model


@torch.no_grad()
def validate(model, loader):
    criterion = torch.nn.BCELoss()  # Usare BCELoss per sigmoid

    model.eval()

    loss = 0
    acc = 0

    for data in loader:
        out = model(data)

        # Calcola la perdita per il batch corrente
        loss += criterion(out, data.y.unsqueeze(1).float()) / len(loader)  # Usa data.y come float per BCELoss

        # Calcola le predizioni usando una soglia di 0.5
        predictions = (out > 0.5).float()

        # Calcola l'accuratezza
        acc += accuracy(predictions, data.y) / len(loader)

    return loss, acc


@torch.no_grad()
def test(model, loader, path, run):
    criterion = torch.nn.BCELoss()  # Usa BCELoss per una singola classe binaria

    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    loss = 0
    acc = 0

    for data in loader:
        out = model(data)

        # Applica sigmoid per ottenere le probabilità
        prob = torch.sigmoid(out)  # Applica sigmoid per problemi binari

        # Calcola la perdita per il batch corrente
        loss += criterion(prob, data.y.unsqueeze(1).float()) / len(loader)  # Data.y deve essere float per BCELoss

        # Converti le probabilità in etichette predette (threshold 0.5)
        preds = (prob > 0.5).int()  # Converte in 0 o 1 con soglia 0.5

        # Calcola l'accuratezza
        acc += accuracy(preds, data.y) / len(loader)

        # Salva le probabilità e le etichette vere
        all_probs.append(prob.cpu().numpy())  # Porta i dati su CPU per sklearn
        all_preds.append(preds.cpu().numpy())  # Porta i dati su CPU per sklearn
        all_labels.append(data.y.cpu().numpy())

    # Concatena i risultati di tutti i batch
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)


    # Calcola le metriche di valutazione
    acc_score = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc_value = auc(fpr, tpr)
    cm = confusion_matrix(all_labels, all_preds)


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

    # Plot della matrice di confusione
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not chimeric', 'Chimeric'],
                yticklabels=['Not chimeric', 'Chimeric'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Run {run})')
    plt.show()

    # Salvataggio dei risultati su file
    with open(path, "a") as f:
        f.write(f"Run: {run}\n")
        f.write(f"Accuracy: {acc_score:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc_value:.4f}\n")
        f.write("\n")  # Linea vuota per separare i risultati
    return loss, acc


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()  # Confronta le previsioni con le etichette reali e calcola la  percentuale di corrispondenza





"""

                            MAIN







chimeric = 'data/100_genes_Datasets_input/dataset_chimeric2.fastq'
not_chimeric = 'data/100_genes_Datasets_input/dataset_non_chimeric.fastq'
chimeric_sequences = estrai_sequenze_geni(chimeric)
not_chimeric_sequences = estrai_sequenze_geni(not_chimeric)
chimeric_data_list = create_graph_data(chimeric_sequences, chimeric=True)
not_chimeric_data_list = create_graph_data(not_chimeric_sequences, chimeric=False)


torch.save(chimeric_data_list, 'dataset/chimeric_dataset_ONE-HOT.pt')
torch.save(not_chimeric_data_list, 'dataset/not_chimeric_dataset_ONE-HOT.pt')
"""

result_path = "results/One-Hote_results_sigmoid.txt"



chimeric_dataset = torch.load("dataset/chimeric_dataset_ONE-HOT.pt", map_location=torch.device('cpu'))
not_chimeric_dataset = torch.load("dataset/not_chimeric_dataset_ONE-HOT.pt", map_location=torch.device('cpu'))
dataset = ConcatDataset([chimeric_dataset, not_chimeric_dataset])

for run in range(1, 6):
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    model = GCN(in_channels, hidden_channels, out_channels)

    model = train(model, train_loader)
    test_loss, test_acc = test(model, test_loader, result_path, run)
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc:.2f}%')