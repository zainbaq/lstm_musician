from tqdm import tqdm
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, random_split
import argparse
from datasets import WaveformDataset
from models.definitions.lstm_model import LSTMModel

torchaudio.set_audio_backend("soundfile")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', nargs="?", default="data/wav/", help='path to .WAV files.')
parser.add_argument('--sequence_length', nargs="?", default=6000, type=int, help='sequence length to train on.')
parser.add_argument('--batch_size', nargs="?", default=128, type=int, help='batch size to train on.')
parser.add_argument('--n_epochs', nargs="?", default=10, type=int, help='number off epochs to train over.')
parser.add_argument('--save_path', nargs="?", default='models/binaries/checkpoint.pt', type=str, help='save path.')

args = parser.parse_args()

WAV_PATH = args.data_path
SEQUENCE_LENGTH = args.sequence_length
BATCH_SIZE = args.batch_size
N_EPOCHS = args.n_epochs

HIDDEN_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    dataset = WaveformDataset(WAV_PATH, SEQUENCE_LENGTH, 'Transformer')

    train_len = int(len(dataset)*0.8)
    lengths = [train_len, len(dataset)-train_len]

    trainset, validset = random_split(dataset, lengths)

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    validloader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = LSTMModel(SEQUENCE_LENGTH, HIDDEN_SIZE, BATCH_SIZE, 2, DEVICE).to(DEVICE)
    criterion = nn.L1Loss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    print(model)

    for epoch in range(N_EPOCHS):
        model.train()
        train_loss = 0.0
        for sample in tqdm(trainloader):
            x, y = sample['sequence'].to(DEVICE), sample['target'].to(DEVICE)

            optimizer.zero_grad()
            if model.model_type == 'Transformer':
                mask = model.generate_square_subsequent_mask(x.size(0)).to(DEVICE)
                output = model(x, mask)
            else:
                output = model(x)

            loss = criterion(output, y)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(trainloader)
        
        model.eval()
        valid_loss = 0.0
        for sample in tqdm(validloader):
            x, y = sample['sequence'].to(DEVICE), sample['target'].to(DEVICE)

            with torch.no_grad():
                if model.model_type == 'Transformer':
                    mask = model.generate_square_subsequent_mask(x.size(0)).to(DEVICE)
                    output = model(x, mask)
                else:
                    output = model(x)

            error = criterion(output, y)
            valid_loss += error.item()
        valid_loss = valid_loss / len(validloader)
        
        model.save_checkpoint(epoch, optimizer, loss.item(), args.save_path)

        print(f'{epoch} - train: {train_loss}, valid: {valid_loss}')

if __name__ == '__main__':
    main()

