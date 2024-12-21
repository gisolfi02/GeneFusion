<p>
  <img src="https://github.com/user-attachments/assets/34e635a8-7d72-436a-948d-5649c4bd35ae", width=200px>
</p>

# GeneFusion

GeneFusion è un progetto dedicato all'analisi delle fusioni geniche utilizzando modelli di deep learningc che ho sviluppato per il mio lavoro di tesi. Il repository contiene implementazioni di modelli basati su GNN per la classificazione delle sequenze geniche e utilizza due diverse tecniche di encoding delle sequenze, DNABERT e One-Hot Encoding.

## Struttura del Repository

- **data/**: Contiene i dataset contententi le sequenze geniche fuse e non fuse.
- **results/**: Include i risultati ottenuti dagli esperimenti dei vari modelli implementati.
- **BERT_model.py**: Script Python che definisce l'addestramento del modello DNABERT utilizzato per la codfica dei kmers delle sequenze genetiche.
- **GNN_model.py**: Script Python che definisce l'architettura dei modelli basati su Graph Neural Network (GNN).
- **main_BERT.py**: Script principale per la costruzione di un dataset compatibile con i modelli GNN codificando i kmers utilizzando il modello DNABERT.
- **main_ONE-HOT.py**: Script principale la costruzione di un dataset compatibile con i modelli GNN codificando i kmers utilizzando la tecnica One-Hot Encoding. 

## Prerequisiti

Assicurati di avere installato le seguenti dipendenze:

- **Torch**
- **Transformers**
- **Torch-Geometric**
- **Toyplot**
- **NumPy**
- **Pandas**
- **Scikit-Learn**
- **Matplotlib**
- **Seaborn**

Puoi installare le dipendenze utilizzando pip:

```bash
pip install torch transformers torch-geometric toyplot numpy pandas scikit-learn matplotlib seaborn
```

## Utilizzo
A seconda della tecnica di encoding che si desidera utilizzare, la pipeline da seguire è diversa.
### Pipeline DNABERT
Addestrare il modello DNABERT eseguendo il seguente comando:
```bash
python BERT_model.py
```
Successivamente runnare il file *main_BERT.py*:
```bash
python main_BERT.py
```
Infine, nel file *GNN_model.py* impostare i seguenti parametri:
- **in_channels**: 768
- **hidden_channels**: 128
- **out_channels**: 1

selezionare il modello desiderato, modificare le righe di codice 333-34 con le seguenti: 
```bash
chimeric_dataset = torch.load("dataset/chimeric_dataset_BERT.pt", map_location=torch.device('cpu'))
not_chimeric_dataset = torch.load("dataset/not_chimeric_dataset_BERT.pt", map_location=torch.device('cpu'))
```
ed eseguire il comando
```bash
python GNN_model.py
```
### Pipeline One-Hot Encoding
Runnare il file *main_ONE-HOT.py*:
```bash
python main_ONE-HOT.py
```
Successivamente, nel file *GNN_model.py* impostare i seguenti parametri:
- **in_channels**: 24
- **hidden_channels**: 16
- **out_channels**: 1

selezionare il modello desiderato, modificare le righe di codice 333-34 con le seguenti:
```bash
chimeric_dataset = torch.load("dataset/chimeric_dataset_ONE-HOT.pt", map_location=torch.device('cpu'))
not_chimeric_dataset = torch.load("dataset/not_chimeric_dataset_ONE-HOT.pt", map_location=torch.device('cpu'))
```
ed eseguire il comando
```bash
python GNN_model.py
```
