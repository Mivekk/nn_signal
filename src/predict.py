import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
from model import LSTMModel

def load_model(model_path):
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, sequence, threshold=0.5):
    model.eval()
    with torch.no_grad():
        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 2)
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

with open('val_data.pkl', 'rb') as f:
    val_data, val_targets = pickle.load(f)

# Get sequence
for data, targets in zip(val_data, val_targets):
    sequence = data
    correct_output = targets
    
    # Make predictions
    threshold = 0.5
    predictions, binary_predictions = predict(model, sequence, threshold)

    plt.figure(figsize=(15, 5))

    # Plot each column of the sequence
    sequence_lines = []
    for col in range(sequence.shape[1]):
        line, = plt.plot(sequence[:, col], label=f'Sequence Column {col+1}')
        sequence_lines.append(line)

    plt.xlabel('Time Step')
    plt.ylabel('Sensor Value')
    plt.ylim(0, 1)
    plt.title('Sequence with Predicted Peaks')

    # Add vertical lines where predictions are 1
    for i, (pred, corr) in enumerate(zip(binary_predictions, correct_output)):
        if pred == 1:
            plt.axvline(x=i, color='r', linestyle='--', alpha=1.0, lw=0.2)
        if corr == 1:
            plt.axvline(x=i, color='g', linestyle='--', alpha=1.0, lw=5)

    # Custom legend lines for the peaks
    predicted_peaks_line = Line2D([0], [0], color='r', lw=2, linestyle='--', alpha=1.0)
    true_peaks_line = Line2D([0], [0], color='g', lw=2, linestyle='--', alpha=1.0)

    # Combine all legend handles and labels
    handles = sequence_lines + [predicted_peaks_line, true_peaks_line]
    labels = [f'Sequence Column {i+1}' for i in range(sequence.shape[1])] + ['Predicted Peaks', 'True Peaks']

    # Create a single combined legend
    plt.legend(handles, labels, loc='upper right')
    plt.show()