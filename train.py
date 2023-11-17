"""
import pandas as pd
import re
import transformers
import gc
import matplotlib.pyplot as plt
device = torch.device('cuda')
import torch.nn.functional as F
"""
import json
import random

with open("intents.json", "r") as f:
    data = json.load(f)
intents = list(data.keys()) # note : add a number to each intent for future training
key_count = len(data)

transformed_data = []
for(key, value) in data.items():
    for t in value :
        transformed_data.append((key, t))
        
labels_map = {}
i = 0
for item in intents:
    labels_map[item] = i
    i += 1
class cfg:
    num_classes = key_count
    epochs=10
    batch_size=10
    lr=1e-5
    max_length=15

from transformers import BertTokenizer , BertModel
import torch
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.labels = [labels_map[label] for (label, text) in transformed_data]
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = cfg.max_length, truncation=True,
                                return_tensors="pt") for (label, text) in transformed_data]      

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
    def print(self) :
        print(self.labels)


import torch.nn as nn

class BertClassifier(nn.Module):

    def __init__(self, dropout=0.2):

        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, cfg.num_classes)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)


        return (linear_output)
    
    def save(self, file):
        torch.save(self.state_dict(), "./model.pth")
    


from torch.optim import Adam
from tqdm import tqdm

def train(model, train_data, val_data, learning_rate, epochs):

    train, val = Dataset(train_data), Dataset(val_data)

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=cfg.batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.lr)

    if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)
                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                
                
                train_label=train_label.to(device)
                output=output.to(device)
                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:

                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                    
            
            print(
                f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')
    model.save('./model.pth')
                  
EPOCHS = cfg.epochs
model = BertClassifier()
model.load_state_dict(torch.load('./model.pth'))
LR = cfg.lr

train_test_split = 0.8
total_samples = len(transformed_data)
train_size = int(train_test_split * total_samples)
random.shuffle(transformed_data)
train_set = transformed_data[:train_size]
test_set = transformed_data[train_size:]

#train(model, train_set, test_set, LR, EPOCHS)
def predict(model, text):
    text_dict = tokenizer(text, padding='max_length', max_length = cfg.max_length, truncation=True, return_tensors="pt")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.cuda()
    mask = text_dict['attention_mask'].to(device)
    input_id = text_dict['input_ids'].squeeze(1).to(device)
        
    with torch.no_grad():
        output = model(input_id, mask)
        print(output)
        label_id = output.argmax(dim=1).item()
        return label_id
    
model.eval()
prediction=predict(model, text='How do i count the number of letters in a string?')

print(prediction)
for i in labels_map:
    if labels_map[i]==prediction:
        print(f"The intent of the given text is {i}")
        intent=i