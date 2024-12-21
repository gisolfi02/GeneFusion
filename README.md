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

- Python 3.x
- PyTorch
- Transformers
- NetworkX
- NumPy
- Pandas

Puoi installare le dipendenze utilizzando pip:

```bash
pip install torch transformers networkx numpy pandas
