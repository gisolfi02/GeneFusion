from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import ConcatDataset
import matplotlib as plt
import seaborn as sns





"""

                        Modello GNN

"""





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
        x = self.conv4(x,edge_index)
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





def train(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    epochs = 200  # Numero di epoche per cui allenare il modello

    model.train() # Imposta il modello in modalità di addestramento

    # Ciclo principale di addestramento per ciascuna epoca
    for epoch in range(epochs + 1):
        # Inizializza le variabili per tenere traccia della perdita totale e dell'accuratezza
        total_loss = 0
        acc = 0
        val_loss = 0
        val_acc = 0

        # Addestramento del modello su mini-batch
        for data in loader:
            optimizer.zero_grad() # Resetto i gradienti dell'ottimizzatore
            out = model(data)  # Passo i dati attraverso il modello
            loss = criterion(out, data.y) # Passo i dati attraverso il modello

            # Aggiorno la perdita totale e l'accuratezza per questa epoca
            total_loss += loss / len(loader)
            acc += accuracy(out.argmax(dim=1), data.y) / len(loader)

            loss.backward() # Calcolo i gradienti per la backpropagation
            optimizer.step()  # Aggiorno i parametri del modello usando l'ottimizzatore

            val_loss, val_acc = validate(model, val_loader) # Validazione del modello sul set di validazione

        # Stampo le metriche ogni 10 epoche per monitorare l'addestramento
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {total_loss:.2f} '
                  f'| Train Acc: {acc*100:>5.2f}% '
                  f'| Val Loss: {val_loss:.2f} '
                  f'| Val Acc: {val_acc*100:.2f}%')

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
def test(model, loader, path, run):
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

    all_preds_binary = (all_preds > 0.5).astype(int)

    # Calcolo delle metriche
    acc_score = accuracy_score(all_labels, all_preds_binary)
    precision = precision_score(all_labels, all_preds_binary)
    recall = recall_score(all_labels, all_preds_binary)
    f1 = f1_score(all_labels, all_preds_binary)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(all_labels, all_preds_binary)

    #Plot della curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
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
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Run {run})')
    plt.show()

    #Salvataggio dei risultati su file
    with open(path, "a") as f:
        f.write(f"Run: {run}\n")
        f.write(f"Accuracy: {acc_score:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write("\n")  # Linea vuota per separare i risultati


    return loss, acc


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()  # Confronta le previsioni con le etichette reali e calcola la  percentuale di corrispondenza





"""

                                MAIN

"""





in_channels = 24
hidden_channels = 256
out_channels = 2

result_path = "results/One-Hot_results.txt"



chimeric_dataset = torch.load("dataset/chimeric_dataset_ONE-HOT.pt", map_location=torch.device('cpu'))
not_chimeric_dataset = torch.load("dataset/not_chimeric_dataset_ONE-HOT.pt", map_location=torch.device('cpu'))


dataset = ConcatDataset([chimeric_dataset[0:10],not_chimeric_dataset[0:10]])

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
    print(f'Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:.2f}%')