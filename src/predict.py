import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model import LSTMModel

from load_data import data, targets

def load_model(model_path):
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, sequence, threshold=0.5):
    model.eval()
    with torch.no_grad():
        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1, seq_len, 1)
        outputs = model(sequence)
        predictions = outputs.squeeze().cpu().numpy()
        binary_predictions = (predictions > threshold).astype(int)  # Convert to binary
        return predictions, binary_predictions

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Path to the saved model
model_path = 'lstm_model.pth'

# Load the model
model = load_model(model_path).to(device)
print("Model loaded from", model_path)

# Get sequence
it = 0
sequence = data[it]
correct_output = targets[it]
 
# Make predictions
threshold = 0.5
predictions, binary_predictions = predict(model, sequence, threshold)

plt.figure(figsize=(15, 5))
sequence_line, = plt.plot(sequence, label='Sequence')
plt.xlabel('Time Step')
plt.ylabel('Sensor Value')
plt.ylim(0, 1)
plt.title('Sequence with Predicted Peaks')

# Add vertical lines where predictions are 1
for i, pred in enumerate(binary_predictions):
    if pred == 1:
        plt.axvline(x=i, color='r', linestyle='--', alpha=1.0, lw=0.2)
    
    if correct_output[i] == 1:
        plt.axvline(x=i, color='g', linestyle='--', alpha=1.0, lw=5.0)

custom_lines = [sequence_line,
                Line2D([0], [0], color='r', lw=2, linestyle='--', alpha=0.5),
                Line2D([0], [0], color='g', lw=2, linestyle='--', alpha=1.0)]

plt.legend(custom_lines, ['Sequence', 'Predicted Peaks', 'True Peaks'])
plt.show()