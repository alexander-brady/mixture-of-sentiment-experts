import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F


# set to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}

def tokenize_text(texts, tokenizer, max_length=128):
    """
    Tokenize a list of texts using the provided tokenizer
    Returns input IDs and attention masks
    """
    encodings = tokenizer(
        list(texts),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encodings['input_ids'], encodings['attention_mask']

# Training function (now uses the passed progress_bar)
def train_epoch(model, data_loader, optimizer, device, epoch, total_epochs, progress_bar):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i, batch in enumerate(data_loader):
        # Unpack and move batch to device
        input_ids, attention_mask, label = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=label
        )

        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == label)
        total_predictions += len(label)

        # Update the main progress bar
        progress_bar.set_postfix({'epoch': f'{epoch + 1}/{total_epochs}',
                                  'train_loss': f'{total_loss / (i + 1):.4f}',
                                  'train_accuracy': f'{(correct_predictions / total_predictions):.4f}'},
                                 refresh=True)
        progress_bar.update(1)  # Increment the progress by one step (batch)

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / total_predictions
    return avg_loss, accuracy

train_df = pd.read_csv('data/training.csv')

train_sentence = train_df['sentence'].tolist()
train_labels = train_df['label'].tolist()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

train_labels_numerical = [label_map[label] for label in train_labels]

train_input_ids, train_attention_masks = tokenize_text(train_sentence, tokenizer)
train_labels_numerical = torch.tensor(train_labels_numerical, dtype=torch.long)

train_dataset = TensorDataset(
    train_input_ids,
    train_attention_masks,
    train_labels_numerical
)

batch_size = 16  

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# load pretrained model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=3
)
# move to device
model = model.to(device)
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training
for epoch in range(1):  # Example: 10 epochs
    with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch + 1}") as progress_bar:
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, epoch, 3, progress_bar)
        print(f"Epoch {epoch + 1} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

# Evaluation
test_df = pd.read_csv('data/test.csv')
test_input_ids, test_attention_masks = tokenize_text(test_df['sentence'].values, tokenizer)

test_dataset = TensorDataset(test_input_ids, test_attention_masks)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

model.eval()
test_preds = []
test_probs = []

with torch.no_grad():
    for batch in test_loader:
        # Unpack and move batch to device
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        # Get predictions
        _, preds = torch.max(logits, dim=1)

        # Store predictions and probabilities
        test_preds.extend(preds.cpu().numpy())
        test_probs.extend(probs[:, 1].cpu().numpy())

test_df['predicted_label'] = test_preds

# Map back to string labels
reverse_label_map = {v: k for k, v in label_map.items()}
test_df['predicted_label'] = test_df['predicted_label'].map(reverse_label_map)

# Add 'id' if not present
if 'id' not in test_df.columns:
    test_df['id'] = test_df.index

# Save to CSV
submission_df = test_df[['id', 'predicted_label']]
submission_df.to_csv("data/predicitions/predicitons_distil-bert.csv", index=False)