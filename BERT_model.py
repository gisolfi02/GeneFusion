import pandas as pd
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import accuracy_score
import numpy as np


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


# Function to process the FASTQ file and create a DataFrame
def process_fastq(file_path, label, k=3):
    """
    Extract sequences from a FASTQ file, kmerize the sequences, and store them in a DataFrame.
    Args:
    - file_path (str): The path to the FASTQ file.
    - label (str): The label to assign to the sequences (e.g., 'chimeric' or 'non-chimeric').
    - k (int): The size of k-mers to generate.

    Returns:
    - df (pd.DataFrame): A DataFrame with two columns: 'kmerized_sequence' and 'label'.
    """
    sequences = []

    # Manually parse the FASTQ file to extract sequences
    with open(file_path, 'r') as file:
        while True:
            file.readline()  # Skip the identifier line
            sequence = file.readline().strip()  # Read the sequence line
            file.readline()  # Skip the '+' line
            file.readline()  # Skip the quality score line

            if not sequence:
                break
            sequences.append(sequence)

    # Apply the kmerize function to each sequence
    kmerized_sequences = [seq2kmer(seq, k) for seq in sequences]

    # Create a DataFrame with kmerized sequences and the label
    df = pd.DataFrame({
        'kmerized_sequence': kmerized_sequences,
        'label': [label] * len(kmerized_sequences)
    })

    return df

# Create the DNADataset class
class DNADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get the index of the highest logit
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy
    }


# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Output directory
    num_train_epochs=15,  # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Strength of weight decay
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,  # Log every 10 steps
    eval_strategy="epoch",  # Evaluate every epoch
    save_steps=500  # Save every 500 steps
)

# Example usage:
k = 6  # Set the k-mer size (you can change this value)

# Process the FASTQ file and generate the DataFrame
df_ch = process_fastq('data/100_genes_Datasets_input/dataset_chimeric2.fastq', 1, k)
df_no_ch = process_fastq('data/100_genes_Datasets_input/dataset_non_chimeric.fastq', 0, k)
df_fused = pd.concat([df_ch, df_no_ch], ignore_index=True)


model_name = "zhihan1996/DNA_bert_6"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Assuming df_fused contains your 6-merized DNA sequences and labels
sequences = df_fused['kmerized_sequence'].tolist()  # List of DNA sequences (6-mers)
labels = df_fused['label'].tolist()  # List of labels (0 or 1)

# Tokenize all sequences at once
encodings = tokenizer(sequences, padding='max_length', truncation=True, max_length=100, return_tensors='pt')


# Convert the labels to tensor format
labels = torch.tensor(labels)

# Create the dataset using the encodings and labels
dataset = DNADataset(encodings, labels)

# Load the pre-trained model for sequence classification (with 2 labels)
model_name = "zhihan1996/DNA_bert_6"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Initialize the Trainer with the classification model
trainer = Trainer(
    model=model,  # The pre-trained model with a classification head
    args=training_args,  # Training arguments
    train_dataset=dataset,  # The dataset for training
    eval_dataset=dataset,  # The dataset for evaluation (can be split if needed)
    tokenizer=tokenizer,  # The tokenizer used
    compute_metrics=compute_metrics  # Pass the custom metrics function
)

# Fine-tune the model
trainer.train()

# Assuming 'model' is your fine-tuned BertForSequenceClassification or BertModel
model.save_pretrained('./bert_model')  # Specify the directory to save the model