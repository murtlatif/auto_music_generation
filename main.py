import librosa
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split
from torch import nn, optim

from config import Config
from constants import note_map
# from data.number_loader import NumberLoader
from data.notes.simple_notes_dataset import SimpleNotes
from util.model_file_manager import load_model
# from preprocess_music import get_dataset, get_label, spectrogram
from util.song_file import get_and_verify_song_path_from_config

from train.trainer import train_one_epoch_transformer
from train.evaluator import validation, test
from model.transformer.transformer_baseline import TransformerModel
from constants import OLD_MCDONALD_NOTES

# def spectro_main():
#     song_path = get_and_verify_song_path_from_config()
#     model_state_path = Config.args.load_model_path

#     chunk_size_s = 0.1
#     overlap = 0
#     waveform, sr = librosa.load(song_path)
#     spectro = spectrogram(waveform, sr, chunk_size_s=chunk_size_s)
#     label = get_label(spectro, model_state_path)

#     # dataset, labels = get_dataset(
#     #     song_path, model_state_path, chunk_size_s, overlap)

#     # notes = [note_map[label] for label in np.argmax(labels, axis=1)]
#     print(label)


def get_transformer_model(hidden_dim: int = 4, num_layers: int = 2):
    dict_size = 8
    transformer_model = TransformerModel(
        input_dict_size=dict_size, output_dict_size=dict_size, hidden_dim=hidden_dim, num_layers=num_layers)

    return transformer_model


def train_transformer_on_notes(model: TransformerModel, epochs: int = 10):
    # Initialize parameters
    batch_size = 8
    train_split_percentage = 0.9

    # Initialize dataset
    old_mcd_indices = [note_map.index(note) + 1 for note in OLD_MCDONALD_NOTES]
    dataset = SimpleNotes(old_mcd_indices)

    train_len = int(len(dataset) * train_split_percentage)
    validation_len = len(dataset) - train_len

    train_set, validation_set = random_split(
        dataset, [train_len, validation_len])
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=True, num_workers=1)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    best_loss = 100

    for i in range(epochs):
        epoch_loss = train_one_epoch_transformer(
            model, criterion, optimizer, train_loader)

        epoch_loss_val = validation(model, criterion, validation_loader)
        # scheduler.step()
        print("epoch: {} train loss: {}".format(i, epoch_loss))
        print("epoch: {} val loss: {}".format(i, epoch_loss_val))
        if epoch_loss_val < best_loss:
            best_loss = epoch_loss_val
            model_name = f"model/cache/transformer_{Config.args.name}_{epoch_loss_val:.5f}.pt"
            torch.save(model.state_dict(), model_name)
    return model_name


if __name__ == '__main__':
    torch.manual_seed(10)

    # Train the model
    model = get_transformer_model()
    model_state_file = train_transformer_on_notes(model)

    # Load the model for testing
    model = get_transformer_model()
    load_model(model_state_file, model)

    # Evaluate the model
    test(model, test_times=10, max_len=4)
