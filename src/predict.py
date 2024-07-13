import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.lines import Line2D
from model import LSTMModel
from load_data import extract_interior_exterior_peaks, get_json_files_content

def load_model(model_path, device):
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    return model

def predict(model, device, sequence, threshold=0.5):
    model.eval()
    with torch.no_grad():
        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 2)
        outputs = model(sequence)
        predictions = outputs.squeeze().cpu().numpy()
        binary_predictions = (predictions > threshold).astype(int)  # Convert to binary

        return binary_predictions

def get_predictions(model, device, data, threshold=0.5):
    predictions = []
    for input in data:
        predictions.append(predict(model, device, input))
        
    return predictions

def load_data(path):
    json_contents = get_json_files_content(path)
    data, targets = extract_interior_exterior_peaks(json_contents)

    return data, targets

def compare_to_manual(og_targets):
   # Load the data
    manual_data_path = "./manual_data/"
    manual_data, manual_targets = load_data(manual_data_path)

    # Initialize counters
    tp, tn, fp, fn = 0, 0, 0, 0
    total = 0
    offset = 25  # Tolerance window for matching

    # Iterate through the sequences
    for i in range(manual_targets.shape[0]):
        total += manual_targets.shape[1]

        # Track matched indices
        matched_manual_indices = deque()

        for j in range(manual_targets.shape[1]):
            # If og_target predicts, try to find a matching manual target within the offset window
            if og_targets[i][j]:
                matched = False
                for k in range(max(j - offset, 0), min(j + offset + 1, manual_targets.shape[1])):
                    if manual_targets[i][k] and k not in matched_manual_indices:
                        tp += 1.0
                        matched_manual_indices.append(k)
                        matched = True
                        break
                if not matched:
                    fp += 1.0

        # After processing, count remaining unmatched manual targets as false negatives
        for j in range(manual_targets.shape[1]):
            if manual_targets[i][j] and j not in matched_manual_indices:
                fn += 1.0

    # Calculate true negatives (total possible matches - (tp + fp + fn))
    tn = total - (tp + fp + fn)

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Print results
    print(f"Accuracy: {100 * (tp + tn) / total:.2f}%")
    print(f"Precision: {100 * precision:.2f}%")
    print(f"Recall: {100 * recall:.2f}%")
    print(f"F1 Score: {100 * 2 * precision * recall / (precision + recall):.2f}%")

    for i in range(manual_targets.shape[0]):
        plt.figure(figsize=(15, 5))

        # Plot each column of the sequence
        sequence_lines = []
        for col in range(manual_data.shape[2]):
            line, = plt.plot(manual_data[i, :, col], label=f'Sequence Column {col+1}')
            sequence_lines.append(line)

        plt.xlabel('Time Step')
        plt.ylabel('Sensor Value')
        plt.ylim(0, 1)
        plt.title('Sequence with Predicted Peaks')

        # Add vertical lines where predictions are 1
        for i, (pred, corr) in enumerate(zip(og_targets[i], manual_targets[i])):
            if pred == 1:
                plt.axvline(x=i, color='r', linestyle='--', alpha=0.5, lw=3)
            if corr == 1:
                plt.axvline(x=i, color='g', linestyle='--', alpha=1.0, lw=5)

        # Custom legend lines for the peaks
        predicted_peaks_line = Line2D([0], [0], color='r', lw=2, linestyle='--', alpha=1.0)
        true_peaks_line = Line2D([0], [0], color='g', lw=2, linestyle='--', alpha=1.0)

        # Combine all legend handles and labels
        handles = sequence_lines + [predicted_peaks_line, true_peaks_line]
        legend_labels = [f'Sequence Column {i+1}' for i in range(manual_data.shape[2])] + ['Predicted Peaks', 'True Peaks']

        # Create a single combined legend
        plt.legend(handles, legend_labels, loc='upper right')
        plt.show()

def main():
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Path to the saved model
    model_path = 'lstm_model_best.pth'

    # Load the model
    model = load_model(model_path, device).to(device)
    print("Model loaded from", model_path)

    og_data, og_targets = load_data("./og_data/")
    model_targets = get_predictions(model, device, og_data)

    compare_to_manual(og_targets)

if __name__ == "__main__":
    main()