import os
from tqdm import tqdm
import torchaudio
import torch
from torch.utils.data import Dataset

class WaveformDataset(Dataset):
    
    
    def __init__(self, data_dir, sequence_len, model_type, transform=None, target_transform=None):
        self.sl = sequence_len
        self.data_dir = data_dir
        self.ftype = '.' + data_dir.split('/')[-2]
        self.model_type = model_type
        self.data_dict = self.load_audio_files(data_dir, '.wav')
        
        self.x, self.y = self.load_data(self.data_dict, self.sl)
        
        self.transform = transform
        self.target_transform = target_transform
    
    def load_audio_files(self, directory, ftype):
        data = {}
        for fname in os.listdir(directory):
            if fname.endswith(ftype):
                fpath = directory + fname
                waveform, sample_rate = torchaudio.load(fpath)
                non_empty_mask = waveform.abs().sum(dim=0).bool() 
                waveform = waveform[:, non_empty_mask]
                data[fname] = {
                    'waveform': waveform, 
                    'sample_rate' : sample_rate
                }
                continue
            else:
                continue
        return data
    
    def load_data(self, data_dict, sequence_len):
        inputs, targets = [], []
        for sample in data_dict.values():
            sequences = torch.stack(self.sequence(sample['waveform'], sequence_len), dim=0)
            target = sequences[1:]
            sequences = sequences[:-1]
            inputs.append(sequences)
            targets.append(target)
        return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)
    
    def sequence(self, t, sequence_len):
        return t.split(sequence_len, -1)[:-1]
                
    def __getitem__(self, idx):
        
        x = self.x[idx]
        y = self.y[idx]
    
        if self.model_type == 'LSTM':
            x = x.transpose(0, 1)
            y = y.transpose(0, 1)
        
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.transform(y)
            
        sample = {'sequence': x, 'target': y}
        
        return sample
    
    def __len__(self):
        return len(self.y)        