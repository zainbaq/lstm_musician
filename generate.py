from train import BATCH_SIZE, HIDDEN_SIZE, SEQUENCE_LENGTH
import torchaudio
import torch
import argparse 
from datasets import WaveformDataset
from models.definitions.lstm_model import LSTMModel

parser = argparse.ArgumentParser()
parser.add_argument('--spark_path', type=str, nargs="?", default='data/wav/dr_ford.wav')
parser.add_argument('--model_path', type=str, nargs="?", default='models/binaries/checkpoint.pt')
parser.add_argument('--n_steps', type=int, nargs="?", default=10)
parser.add_argument('--output_path', type=str, nargs="?", default='outputs/raw/song.wav')

args = parser.parse_args()

def main():
    waveform, sample_rate = torchaudio.load(args.spark_path)
    waveform.shape, sample_rate

    model = LSTMModel(SEQUENCE_LENGTH, HIDDEN_SIZE, BATCH_SIZE, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    model, _ = model.load_checkpoint(args.model_path, model, optimizer)
    dataset = WaveformDataset('data/wav/', model.sequence_length, model.model_type)

    wf = torch.stack(dataset.sequence(waveform, model.sequence_length), dim=0)

    song = model.forecast(wf, args.n_steps).detach()

    if args.output_path != None:
        torchaudio.save(args.output_path, song, 48000)
        print(f'Saved song to {args.output_path}')

if __name__ == '__main__':
    main()