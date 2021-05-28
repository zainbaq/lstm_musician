import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    
    def __init__(self, seq_len=6000, hidden_size=128, batch_size=128, n_layers=2, device='cpu'):
        super().__init__()
        self.model_type = 'LSTM'
        self.batch_size = batch_size
        self.sequence_length = seq_len
        self.device = device

        self.lstm = nn.LSTM(
            seq_len,
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

        hidden_state = hidden_state.to(device)
        cell_state = cell_state.to(device)
            
        self.hidden = (hidden_state, cell_state)
        
    def forward(self, x):
        
        # LSTM layer
        x, h = self.lstm(x, self.hidden)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)     
        
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
#         print(x.shape)
        return x

    def forecast(self, wf, n_steps):
        history = wf[:self.batch_size].unsqueeze(0)
        for i in range(n_steps):
            next_step = self.forward(history[-1].to(self.device))
            history = torch.cat((history, next_step.cpu().unsqueeze(0)), dim=0)
        return history.view(2, -1)

    def save_checkpoint(self, epoch, optimizer, loss, f_path):
        torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, f_path)
                
    def load_checkpoint(self, checkpoint_path, model, optimizer):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, optimizer