import torch
import torch.nn as nn

# def save_checkpoint(checkpoint):


class LSTMModel(nn.Module):
    
    def __init__(self, num_features, seq_len, hidden_size, batch_size, n_layers, device):
        super().__init__()
        self.model_type = 'LSTM'
        self.batch_size = batch_size
        self.lstm = nn.LSTM(
            num_features,
            hidden_size,
            batch_first=True,
            num_layers=n_layers
        )
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, seq_len)
        self.tanh = nn.Tanh()
        
        hidden_state = torch.zeros(n_layers, batch_size, hidden_size)
        cell_state = torch.zeros(n_layers, batch_size, hidden_size)
            
        self.hidden = (hidden_state.to(device), cell_state.to(device))
        
    def forward(self, x):
        
        # LSTM layer
        x, h = self.lstm(x, self.hidden)

        # Fully Connected layer
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)     
        
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x