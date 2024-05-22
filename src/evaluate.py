import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluation
def evaluate_model(model, dataloader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.unsqueeze(-1)
            outputs = model(data)
            predictions.append(outputs.squeeze().cpu().numpy())
            actuals.append(targets.cpu().numpy())
            
    print_evaluation_metrics(np.concatenate(predictions), np.concatenate(actuals))

def print_evaluation_metrics(predictions, actuals, threshold=0.5):
    # Converting predictions to binary form (0 or 1) using a threshold
    binary_predictions = (predictions > threshold).astype(int)

    # Example evaluation metrics
    accuracy = accuracy_score(actuals, binary_predictions)
    precision = precision_score(actuals, binary_predictions, average='macro', zero_division=1)
    recall = recall_score(actuals, binary_predictions, average='macro', zero_division=1)
    f1 = f1_score(actuals, binary_predictions, average='macro', zero_division=1)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')