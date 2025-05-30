import torch, time, os
import torch.nn as nn
import torch.optim as optim

from typing import Optional, Union
from tqdm import tqdm


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: Union[str, torch.device],
    num_epochs: int = 3,
    learning_rate: float = 1e-5,
):
    '''
    Train a neural network model.
    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader for the training dataset.
        val_loader: Optional DataLoader for the validation dataset.
        device: Device to run the training on (e.g., 'cpu' or 'cuda').
        num_epochs: Number of epochs to train the model.
        learning_rate: Learning rate for the optimizer.
    '''
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=float(learning_rate))

    for epoch in range(num_epochs):
        model.train()
        
        total_loss = 0
        tq = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in tq:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            tq.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
        if val_loader:
            validate(model, val_loader, device)
        
        
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Union[str, torch.device]
):    
    '''
    Validate the model on the validation set.
    Args:
        model: The neural network model to be validated.
        dataloader: DataLoader for the validation dataset.
        device: Device to run the validation on (e.g., 'cpu' or 'cuda').
    '''
    model.eval()
    total_mae = 0

    tq = tqdm(dataloader, desc="Validation")
    with torch.inference_mode():
        for batch in tq:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
            mae = torch.mean(torch.abs(preds.float() - labels.float()))
            total_mae += mae.item()
            tq.set_postfix(mae=mae.item() / labels.size(0))
            
    avg_mae = total_mae / len(dataloader)
    score = 0.5 * (2 - avg_mae)
    print(f"Validation | Score: {score:.4f} | MAE: {avg_mae:.4f}")
    
    
def predict(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: Union[str, torch.device],
    label_map: Optional[dict] = None
):
    '''
    Evaluate a neural network model on a dataset.
    
    Args:
        model: The neural network model to be evaluated.
        dataloader: DataLoader for the dataset to evaluate on.
        label_map: Optional mapping from label indices to label names.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    predictions = []
    
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            outputs = logits.argmax(dim=-1)
            predictions.extend(outputs.cpu().numpy())
            
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    with open(f'results/{timestamp}.csv', 'w') as f:
        f.write("id,label\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{label_map[pred]}\n")
    
    os.mkdir('results', exist_ok=True)
    print(f"Evaluation complete. Predictions saved to 'results/{timestamp}.csv'.")