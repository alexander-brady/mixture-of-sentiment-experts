import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
# print("CUDA available:", torch.cuda.is_available())

# Training data
import os
print(os.getcwd())
trainDF = pd.read_csv('data/training.csv')
trainDF.columns = ['id', 'text', 'label']

X_train, y_train = (trainDF['text'], trainDF['label'])

# Load pre-trained SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# print("Encoding training data...")
X_train_embeds = model.encode(X_train.tolist(), show_progress_bar=True, convert_to_numpy=True)

# Train a logistic regression classifier
clf = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced'
)
clf.fit(X_train_embeds, y_train)

# Evaluate the model
# print("Evaluating the model...")
testDF = pd.read_csv('data/test.csv')
testDF.columns = ['id', 'text']

# Predictions 
X_val = testDF['text'].to_list()
X_val_embeds = model.encode(X_val, show_progress_bar=True, convert_to_numpy=True)
predictions = clf.predict(X_val_embeds)
predictions = pd.DataFrame({'id': range(len(predictions)), 'label': predictions})
predictions.to_csv('data/predictions/predictions_model1.csv', index=False)
