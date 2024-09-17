import torch
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification, PreTrainedTokenizer
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import collections
from itertools import product
import os

def estrai_sequenze_geni(file_fastq):
    sequenze_geni = []
    k = 0
    with open(file_fastq, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            sequence = lines[i + 1].strip()
            sequenze_geni.append(sequence)
            k += 1
            if(k == 960):
                break
    return sequenze_geni

def get_kmer(sequence, k=4):
    kmers = []
    for i in range(0, len(sequence)):
        if len(sequence[i:i + k]) != k:
            continue
        kmers.append(sequence[i:i + k])
    return kmers


def create_vocab_file(vocab_path, len_kmer=4, add_n=True):
    """
    Create a vocabulary file for DNA sequences with k-mers.

    Parameters:
    vocab_path (str): Path to save the vocabulary file.
    len_kmer (int): Length of k-mers to include in the vocabulary.
    add_n (bool): Whether to include 'N' in the vocabulary.
    """
    # Define the alphabet
    bases = ['A', 'C', 'G', 'T', 'N'] if add_n else ['A', 'C', 'G', 'T']

    # Generate all possible k-mers
    kmers = [''.join(kmer) for kmer in product(bases, repeat=len_kmer)]

    # Special tokens
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

    # Write to vocab file
    with open(vocab_path, 'w') as vocab_file:
        for token in special_tokens:
            vocab_file.write(f'{token}\n')
        for kmer in kmers:
            vocab_file.write(f'{kmer}\n')

# Tokenizer Creation
class DNABertTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, **kwargs):
        """
        Custom DNA BERT Tokenizer for handling DNA sequences and converting them into k-mers.

        :param vocab_file: Path to the pre-created vocabulary file.
        """
        self.vocab_name = 'vocab_kmer_4'  # Nome del file del vocabolario
        self.vocab_path = f'./{self.vocab_name}.txt'

        # Load the vocabulary into memory
        self.vocab = self._load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict((i, t) for t, i in self.vocab.items())

        # Initialize the superclass with the vocab file
        super().__init__(vocab_file=vocab_file, **kwargs)

        # Define special tokens
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"

        # Add special tokens to the tokenizer
        self.add_special_tokens({
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "mask_token": self.mask_token,
        })

    @property
    def vocab_size(self):
        """
        Returns the size of the vocabulary.
        """
        return len(self.vocab)

    def _load_vocab(self, vocab_file):
        """
        Load the vocabulary from a file into a dictionary.
        """
        vocab = collections.OrderedDict()
        with open(vocab_file, 'r') as f:
            for idx, token in enumerate(f.readlines()):
                vocab[token.strip()] = idx
        return vocab

    def _tokenize(self, text):
        """
        Tokenize the input DNA sequence by splitting it into k-mers.
        """
        kmer_size = len(next(iter(self.vocab)))  # Determine the k-mer length from the vocab
        return [text[i:i+kmer_size] for i in range(0, len(text) - kmer_size + 1)]

    def _convert_token_to_id(self, token):
        """
        Convert a token (k-mer) into an ID using the vocab.
        """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """
        Convert an ID back into a token (k-mer).
        """
        return self.ids_to_tokens.get(index, self.unk_token)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a single sequence by adding special tokens.
        Sequence: [CLS] X [SEP]
        """
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep

    def get_vocab(self):
        """
        Return the vocabulary as a dictionary mapping tokens to IDs.
        """
        return self.vocab

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the tokenizer vocabulary to the specified directory.
        """
        vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_name + ".txt")
        with open(vocab_file, 'w') as f:
            for token, idx in self.vocab.items():
                f.write(f"{token}\n")
        return (vocab_file,)


class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Tokenize the sequence
        encoded = self.tokenizer(
            sequence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Extract input_ids and attention_mask
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label)
        }

chimeric = 'data/100_genes_Datasets_input/dataset_chimeric2.fastq'
not_chimeric = 'data/100_genes_Datasets_input/dataset_non_chimeric.fastq'

chimeric_sequences = estrai_sequenze_geni(chimeric)
not_chimeric_sequences = estrai_sequenze_geni(not_chimeric)



# Run this script to create the vocab file
vocab_path = './vocab_kmer_4.txt'
create_vocab_file(vocab_path, len_kmer=4, add_n=True)

print(f'Vocabulary created and saved to {vocab_path}')

tokenizer = DNABertTokenizer(vocab_file='./vocab_kmer_4.txt')

chimeric_kmers_sequences = []
not_chimeric_kmers_sequences = []

for sequence in chimeric_sequences:
    kmers = get_kmer(sequence)
    chimeric_kmers_sequences.append(kmers)

for sequence in not_chimeric_kmers_sequences:
    kmers = get_kmer(sequence)
    not_chimeric_kmers_sequences.append(kmers)

sequences = chimeric_kmers_sequences + not_chimeric_kmers_sequences
labels = [1] * len(chimeric_sequences) + [0] * len(not_chimeric_sequences)

BERT_dataset = DNADataset(sequences, labels, tokenizer, max_length=128)
dataloader = DataLoader(BERT_dataset, batch_size=64, shuffle=True)

config = BertConfig(
    vocab_size=len(tokenizer.vocab),  # The size of your custom DNA vocab
    hidden_size=768,  # Hidden layer size
    num_labels=2
)

# Initialize the BERT model
BERT_model = BertForSequenceClassification(config)
optimizer = torch.optim.AdamW(BERT_model.parameters(), lr=0.01)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BERT_model.to(device)

print("Addestramento BERT model")
# Loop per addestramento
for epoch in range(500):
    BERT_model.train()  # Imposta il modello in modalità di addestramento
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # Forward pass per calcolare gli output, include labels
        outputs = BERT_model(input_ids, attention_mask=attention_mask, labels=labels)

        # Estrai la loss dagli output del modello
        loss = outputs.loss

        # Estrai le predizioni dal modello
        logits = outputs.logits

        # Ottieni le predizioni con il valore massimo (classe predetta)
        predictions = torch.argmax(logits, dim=-1)

        # Aggiorna il conteggio delle predizioni corrette e totali
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

        # Backward pass per aggiornare i pesi
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Calcolo dell'accuracy
    accuracy = correct_predictions / total_predictions
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {accuracy:.4f}")

save_directory = "./bert_model"
BERT_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Modello salvato nella directory: {save_directory}")