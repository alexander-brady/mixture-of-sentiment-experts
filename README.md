# Mixture-of-Experts for Sentiment Classification

> Alexander Brady, Adam Suma, Cedric Girardin, Diego Calasso

This repository contains the source code for the team _The Sentimentalist_'s project on sentiment classification using a Mixture-of-Experts (MoE) model for the course Computational Intelligence Lab (2025).

## Setup

1. Download the dataset from [here](https://www.kaggle.com/competitions/ethz-cil-text-classification-2025/overview). Ensure you have the `train.csv` and `test.csv` files in the `data/` directory, or update the file paths in `config.yaml/dataloaders`.
2. Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Run the training and evaluation pipeline:
   ```bash
    python src/run.py --model moe # Specify the model type (default is moe)
    ```

**SLURM**

Optionally, you can run the training script using SLURM for distributed training. Ensure you have SLURM configured on your system and use the following command:
```bash
sbatch eval.sh <model_name>
```
This will create a temporary directory for the SLURM job, create an environment, and run the training script with the specified model type.

**Model Types**

The currently available model baselines are:
- **moe**: Mixture-of-Experts
- **deberta-v3-base**: finetuned microsoft/deberta-v3-base
- **deberta-v3-large**: finetuned microsoft/deberta-v3-large
- **sentence-transformer**: all-MiniLM-L6-v2 embeddings + Logistic Regression
- **distil-bert**: Finetuned distilbert-base-uncased

## Notebooks
- `data_analysis.ipynb` Analyzes the dataset and applies topic modeling on it.
