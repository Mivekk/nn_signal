import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import LSTMModel, hidden_size, num_layers
from load_data import data, targets

# Set the random seed for reproducibility
seed = 39
# NumPy
np.random.seed(seed)
# PyTorch
torch.manual_seed(seed)

# If you are using GPUs
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Split data into training and test sets
train_data, val_data, train_targets, val_targets = train_test_split(data, targets, test_size=0.2, random_state=seed)

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
val_dataset = SensorDataset(val_data, val_targets)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = LSTMModel().to(device)

# Calculate class weights
num_zeros = (train_targets == 0).sum()
num_ones = (train_targets == 1).sum()
pos_weight = num_zeros / num_ones

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

def train_one_epoch(model, dataloader, criterion, optimizer, grad_clip=1.0):
    epoch_train_losses = []
    model.train()
    for data, targets in dataloader:
        # Add feature dimension (seq_len, batch_size, input_size)
        data, targets = data.to(device).unsqueeze(-1), targets.to(device).unsqueeze(-1)
        
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        epoch_train_losses.append(loss.item())

    return np.mean(epoch_train_losses)

def validate_one_epoch(model, dataloader, criterion):
    epoch_val_losses = []
    model.eval()
    val_predictions, val_actuals = [], []
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device).unsqueeze(-1), targets.to(device).unsqueeze(-1)
            outputs = model(data)
            loss = criterion(outputs, targets)
            epoch_val_losses.append(loss.item())
            val_predictions.append(outputs.cpu().numpy())
            val_actuals.append(targets.cpu().numpy())
    
    val_loss = np.mean(epoch_val_losses)
    val_predictions = (np.concatenate(val_predictions) > 0.5).astype(int).reshape(-1)
    val_actuals = np.concatenate(val_actuals).reshape(-1)

    return val_loss, val_predictions, val_actuals


# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_predictions, val_actuals = validate_one_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        val_precision = precision_score(val_actuals, val_predictions, average='binary', zero_division=1)
        val_recall = recall_score(val_actuals, val_predictions, average='binary', zero_division=1)
        val_f1 = f1_score(val_actuals, val_predictions, average='binary', zero_division=1)
        
        print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}\n')

    return train_losses, val_losses

train_losses, val_losses = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10)

# Plot the train and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Train and Validation Loss Over Epochs\nHidden size: {hidden_size}, Number of layers: {num_layers}')
plt.legend()
plt.show()

model_path = "lstm_model.pth"
torch.save(model.state_dict(), model_path)
print("Model saved to", model_path)
