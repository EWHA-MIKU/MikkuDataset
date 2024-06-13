# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from datasets import load_metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

###############################
# Load the training data 
data_path = "/mnt/aix21006/data/gender_labeled/1500/gender_labeled_except_predict_1500_train.csv"
data = pd.read_csv(data_path, encoding="utf-8")
data['combined'] = data['ko'] + " " + data['figure'].astype(str)

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['combined'].tolist(),
    data['gender_real'].tolist(),
    test_size=0.1  # Example: 10% as validation data
)

# Get unique labels and create a label map
unique_labels = list(set(train_labels))
label_to_id = {label: i for i, label in enumerate(unique_labels)}

###############################
# Load the tokenizer and the model from Hugging Face
# model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
# model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
model_name = "MoritzLaurer/bge-m3-zeroshot-v2.0"
# "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_labels)).to(device)

# Tokenize the training and validation texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
train_dataset = CustomDataset(train_encodings, [label_to_id[label] for label in train_labels])
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
val_dataset = CustomDataset(val_encodings, [label_to_id[label] for label in val_labels])

# Metric functions - 추가 
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average='macro', zero_division=1)
    rec = recall_score(labels, preds, average='macro', zero_division=1)
    f1 = f1_score(labels, preds, average='macro', zero_division=1)
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
    }

########################################
# Training arguments for the Trainer
# 나중에 그리드 서치 추가할지도.. 
training_args = TrainingArguments(
    output_dir='./results/results5_1',           
    num_train_epochs=5,
    per_device_train_batch_size=16,     
    learning_rate=3e-5,                 
    logging_dir='./logs/logs5_1',             
    logging_steps=250,                  
    save_strategy="steps",              
    evaluation_strategy="steps",        
    eval_steps=500,                     
    save_total_limit=2,                 
    load_best_model_at_end=True,        
    metric_for_best_model="eval_loss",       
    greater_is_better=False             
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Start training
trainer.train()

# Save the trained model and tokenizer
model_save_path = "./saved_model5_1"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Load the test data
test_data_path = "/mnt/aix21006/data/gender_labeled/1500/gender_labeled_except_predict_1500_test.csv"
test_data = pd.read_csv(test_data_path, encoding="utf-8")
test_data['combined'] = test_data['ko'] + " " + test_data['figure'].astype(str)
test_texts = test_data['combined'].tolist()
test_labels = test_data['gender_real'].tolist()

# Tokenize the test texts
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)
test_dataset = CustomDataset(test_encodings, [label_to_id[label] for label in test_labels])

# Predict on the test dataset
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=-1)

# Convert numerical predictions back to string labels
pred_labels_str = [unique_labels[label] for label in pred_labels]

# Create a new DataFrame with the original test data and the predictions
test_results = test_data.copy()
test_results['gender_pred'] = pred_labels_str

# # Calculate the metrics
# accuracy = accuracy_score(test_dataset.labels, pred_labels)
# precision = precision_score(test_dataset.labels, pred_labels, average='macro', zero_division=1)
# recall = recall_score(test_dataset.labels, pred_labels, average='macro', zero_division=1)
# f1 = f1_score(test_dataset.labels, pred_labels, average='macro', zero_division=1)

# # Print the metrics
# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1: {f1}")

# Save the new DataFrame to a CSV file
test_results.to_csv("결과 저장 경로!", encoding='utf-8-sig', index=False)

#############################################
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns  
import matplotlib.pyplot as plt

# Load the labeled data
data = test_results

# Extract the ground truth labels
y_true_kr = data['gender_real']

# Extract the predicted labels
y_pred_kr = data['gender_pred']

# Calculate micro F1 score
micro_f1_kr = f1_score(y_true_kr, y_pred_kr, average='micro')

# Calculate macro F1 score
macro_f1_kr = f1_score(y_true_kr, y_pred_kr, average='macro')

# Calculate precision and recall
precision_kr1 = precision_score(y_true_kr, y_pred_kr, average='micro', zero_division=1)
recall_kr1 = recall_score(y_true_kr, y_pred_kr, average='micro', zero_division=1)

# Calculate precision and recall
precision_kr2 = precision_score(y_true_kr, y_pred_kr, average='macro', zero_division=1)
recall_kr2 = recall_score(y_true_kr, y_pred_kr, average='macro', zero_division=1)

# Calculate accuracy
accuracy_kr = accuracy_score(y_true_kr, y_pred_kr)

# Print the scores
print("\nKOREAN")
print("Micro F1 score:", micro_f1_kr)
print("Macro F1 score:", macro_f1_kr)
print("Micro Precision:", precision_kr1)
print("Macro Precision:", precision_kr2)
print("Micro Recall:", recall_kr1)
print("Macro Recall:", recall_kr2)
print("Accuracy:", accuracy_kr)

# Calculate the confusion matrix
cm_kr = confusion_matrix(y_true_kr, y_pred_kr)

# Print the confusion matrix
print("\nConfusion Matrix:")
print(cm_kr)

# Visualize the confusion matrix using Seaborn
plt.figure(figsize=(8,6))
sns.heatmap(cm_kr, annot=True, cmap="YlGnBu", fmt='g', xticklabels=['M', 'F', 'F&M'], yticklabels=['M', 'F', 'F&M'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('ZS') 

# Save the visualization to a file
plt.savefig("./zero3_2.png", bbox_inches='tight')

# Optionally: close the plot if you don't want to show it in a GUI-based environment
plt.close()
