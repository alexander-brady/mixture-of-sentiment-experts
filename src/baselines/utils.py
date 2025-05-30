import yaml
from typing import Optional
from datasets import load_dataset
from torch.utils.data import DataLoader


def load_config() -> dict:
    """
    Load configuration from a YAML file.

    Returns:
        dict: Configuration parameters as a dictionary.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    return config


def load_dataloaders(
    tokenizer,
    validation_split: float = 0,
    batch_size: int = 16,
    max_length: int = 384,
    train_set: str = 'data/training.csv',
    test_set: str = 'data/test.csv',
    label_map: Optional[dict] = None,
):
    '''
    Load training and test datasets, tokenize them, and prepare DataLoaders.
    
    Args:
        tokenizer: Tokenizer to be used for tokenizing the text data.
        validation_split (float): Proportion of the training set to be used for validation. If 0, return None for Val_loader.
        batch_size (int): Batch size for the DataLoader.
        max_length (int): Maximum length of the tokenized sequences.
        label_map (Optional[dict]): Mapping from string labels to integer values.
    
    Returns:
        tuple: Training, Validation, Test Dataloaders
    '''    
    def tokenize_fn(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=max_length)

    def cast_label_fn(batch):
        batch["label"] = [label_map[label] for label in batch["label"]]

    dataset = load_dataset('csv', data_files={
        'train': train_set,
        'test': test_set
    })
    dataset = dataset.map(tokenize_fn, batched=True)

    dataset['train'].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataset['test'].set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    if label_map:
        dataset['train'] = dataset['train'].map(cast_label_fn, batched=True)
    
    if validation_split > 0:
        split_ds = dataset['train'].train_test_split(test_size=validation_split)
        
        train_loader = DataLoader(split_ds['train'], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(split_ds['test'], batch_size=batch_size, shuffle=False)
    
    else:
        train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
        val_loader = None
    test_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader