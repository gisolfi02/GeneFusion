# GeneFusion

GeneFusion is a project dedicated to the analysis of gene fusions using deep learning models, developed as part of my thesis work. The repository contains implementations of GNN-based models for classifying gene sequences and utilizes two different sequence encoding techniques: DNABERT and One-Hot Encoding.

## Repository Structure

- **data/**: Contains datasets with fused and non-fused gene sequences.
- **results/**: Includes results obtained from experiments with various implemented models.
- **BERT_model.py**: Python script defining the training of the DNABERT model used for k-mer encoding of genetic sequences.
- **GNN_model.py**: Python script defining the architecture of Graph Neural Network (GNN) models.
- **main_BERT.py**: Main script for constructing a dataset compatible with GNN models by encoding k-mers using the DNABERT model.
- **main_ONE-HOT.py**: Main script for constructing a dataset compatible with GNN models by encoding k-mers using the One-Hot Encoding technique.

## Prerequisites

Make sure you have installed the following dependencies:

- **Torch**
- **Transformers**
- **Torch-Geometric**
- **Toyplot**
- **NumPy**
- **Pandas**
- **Scikit-Learn**
- **Matplotlib**
- **Seaborn**

You can install the dependencies using pip:

```bash
pip install torch transformers torch-geometric toyplot numpy pandas scikit-learn matplotlib seaborn
```

## Usage
Depending on the encoding technique you wish to use, follow the respective pipeline.

### DNABERT Pipeline
Train the DNABERT model by running the following command:
```bash
python BERT_model.py
```
Then, run the *main_BERT.py* file:
```bash
python main_BERT.py
```
Finally, in the *GNN_model.py* file, set the following parameters:
- **in_channels**: 768
- **hidden_channels**: 128
- **out_channels**: 1

Select the desired model and modify lines 333-334 with the following:
```bash
chimeric_dataset = torch.load("dataset/chimeric_dataset_BERT.pt", map_location=torch.device('cpu'))
not_chimeric_dataset = torch.load("dataset/not_chimeric_dataset_BERT.pt", map_location=torch.device('cpu'))
```
and execute the command:
```bash
python GNN_model.py
```

### One-Hot Encoding Pipeline
Run the *main_ONE-HOT.py* file:
```bash
python main_ONE-HOT.py
```
Then, in the *GNN_model.py* file, set the following parameters:
- **in_channels**: 24
- **hidden_channels**: 16
- **out_channels**: 1

Select the desired model and modify lines 333-334 with the following:
```bash
chimeric_dataset = torch.load("dataset/chimeric_dataset_ONE-HOT.pt", map_location=torch.device('cpu'))
not_chimeric_dataset = torch.load("dataset/not_chimeric_dataset_ONE-HOT.pt", map_location=torch.device('cpu'))
```
and execute the command:
```bash
python GNN_model.py
