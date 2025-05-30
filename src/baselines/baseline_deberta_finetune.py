import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# ------------ Config ------------
SEED = 42
TRAIN_CSV = 'data/training.csv'
TEST_CSV = 'test.csv'
LABEL2ID = {'negative': 0, 'neutral': 1, 'positive': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
# --------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(batch, tokenizer):
    tokens = tokenizer(
        batch['sentence'],
        max_length=128,
        truncation=True,
        padding=False
    )
    if 'label' in batch:
        tokens['labels'] = [LABEL2ID[l] for l in batch['label']]
    return tokens


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    mae = mean_absolute_error(labels - 1, preds - 1)
    score = 0.5 * (2 - mae)
    return {'mae': mae, 'score': score}


def train_and_predict(model_size='base'):
    """
    model_size: 'base' or 'large'
    """
    set_seed(SEED)
    print('Using GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

    if model_size == 'large':
        checkpoint = 'microsoft/deberta-v3-large'
        out_csv = 'prediction_deberta_large.csv'
    else:
        checkpoint = 'microsoft/deberta-v3-base'
        out_csv = 'prediction_deberta_base.csv'

    # Load data
    df = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # Stratified split train/val
    train_df, val_df = train_test_split(
        df, test_size=0.1, stratify=df['label'], random_state=SEED
    )

    # Create Datasets
    ds = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(val_df),
        'test': Dataset.from_pandas(df_test)
    })

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    for split in ['train', 'validation', 'test']:
        remove_cols = ['sentence', 'label'] if split != 'test' else ['sentence']
        ds[split] = ds[split].map(
            lambda b: tokenize(b, tokenizer),
            batched=True,
            remove_columns=remove_cols
        )

    # Prepare model and trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    data_collator = DataCollatorWithPadding(tokenizer)
    args = TrainingArguments(
        output_dir=f'out_{model_size}',
        num_train_epochs=3,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='mae',
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()
    model_dir = f"models/deberta-v3-{model_size}"
    trainer.save_model(model_dir)

    # Predict on test
    preds = trainer.predict(ds['test']).predictions
    preds = np.argmax(preds, axis=1)

    # Save CSV
    df_test['label'] = [ID2LABEL[p] for p in preds]
    df_test[['id', 'label']].to_csv(out_csv, index=False)
    print(f'Results saved to {out_csv}')



# Example usage:
if __name__ == '__main__':
    train_and_predict('large')
    #train_and_predict('base')
