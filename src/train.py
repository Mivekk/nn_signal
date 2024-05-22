import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from model import LSTMModel
from load_data import data, targets
from evaluate import evaluate_model

# Split data into training and test sets
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2, random_state=42)

# Dataset and DataLoader
class SensorDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

train_dataset = SensorDataset(train_data, train_targets)
test_dataset = SensorDataset(test_data, test_targets)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = LSTMModel()

# Calculate class weights
num_zeros = (train_targets == 0).sum()
num_ones = (train_targets == 1).sum()
pos_weight = num_zeros / num_ones * 4

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for data, targets in dataloader:
            # Add feature dimension (seq_len, batch_size, input_size)
            data, targets = data.unsqueeze(-1), targets.unsqueeze(-1)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

train_model(model, train_dataloader, criterion, optimizer, num_epochs=10)

model_path = "lstm_model.pth"
torch.save(model.state_dict(), model_path)
print("Model saved to", model_path)

# Example evaluation
evaluate_model(model, test_dataloader)