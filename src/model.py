import torch
import torch.nn as nn

hidden_size = 300
num_layers = 6

# Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Get the LSTM outputs
        out, _ = self.lstm(x, (h0, c0))
        
        # Pass the LSTM outputs through a fully connected layer
        out = self.fc(out)
        return out