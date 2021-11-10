import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import random_split

from config import Config
from data.dataset.music_dataset import MusicDataset
from model.transformer.transformer_baseline import TransformerModel
from train.evaluator import evaluate_note_sequence
from train.trainer import train_transformer
from util.constants import PAD_TOKEN, Songs
from util.model_file_manager import load_model
from util.random_substring import get_random_substrings


def get_transformer_model(hidden_dim: int = 512, num_layers: int = 6):
    dict_size = 8
    transformer_model = TransformerModel(
        input_dict_size=dict_size, output_dict_size=dict_size, hidden_dim=hidden_dim, num_layers=num_layers)

    return transformer_model


def get_music_data_loaders(songs: list[str], batch_size: int, max_sequence_len: int, train_split_percentage: float = 0.9):
    # Default max sequence length to the maximum song length
    if max_sequence_len is None:
        max_sequence_len = max([len(song) for song in songs])

    dataset = MusicDataset(songs, max_sequence_length=max_sequence_len)

    train_len = int(len(dataset) * train_split_percentage)
    validation_len = len(dataset) - train_len

    train_set, validation_set = random_split(dataset, [train_len, validation_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, validation_loader


def get_song_snippets(song: str, len_snippets: int, num_snippets: int):
    snippets = get_random_substrings(song, len_snippets, num_snippets, step=4)
    return snippets


def train_transformer_on_notes(model: TransformerModel, epochs: int = 10, batch_size: int = 32, print_status: bool = True, save_best_model: bool = True):
    # Initialize parameters
    train_split_percentage = 0.9
    song_sample_length = 5

    # Initialize dataset
    all_songs = []
    all_songs.extend(get_song_snippets(Songs.JINGLE_BELLS, len_snippets=song_sample_length, num_snippets=128))
    all_songs.extend(get_song_snippets(Songs.ODE_TO_JOY, len_snippets=song_sample_length, num_snippets=64))
    all_songs.extend(get_song_snippets(Songs.TWINKLE_TWINKLE, len_snippets=song_sample_length, num_snippets=32))
    all_songs.extend(get_song_snippets(Songs.OLD_MCDONALD, len_snippets=song_sample_length, num_snippets=32))
    all_songs.extend(get_song_snippets(Songs.ROW_YOUR_BOAT, len_snippets=song_sample_length, num_snippets=16))
    all_songs.extend(get_song_snippets(Songs.HAPPY_BIRTHDAY, len_snippets=song_sample_length, num_snippets=16))

    train_loader, validation_loader = get_music_data_loaders(all_songs, batch_size, song_sample_length-1)

    # Initialize training objects
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    train_losses, validation_losses, best_model_file = train_transformer(model, train_loader, validation_loader, optimizer,
                                                                         criterion, epochs, print_status, save_best_model)

    # TODO: Plot the training and validation loss
    # HERE #

    return best_model_file


if __name__ == '__main__':

    print('Running main.py with arguments:', Config.args)
    torch.manual_seed(Config.args.seed)

    # Get the initial model
    model = get_transformer_model()

    # Load model state from file if state is given
    load_model_path = Config.args.load_model_path
    if load_model_path:
        load_model(load_model_path, model)

    # Train the model if the train flag is present
    if Config.args.train:
        model_state_file = train_transformer_on_notes(
            model, epochs=Config.args.epochs, batch_size=Config.args.batch_size)

    # Interactive evaluation with user-input sequences
    if Config.args.interactive:
        try:
            while True:
                input_sequence = input('Enter input sequence: ')
                processed_output = evaluate_note_sequence(model, input_sequence)

                print(f'Input: {input_sequence}')
                print(f'Output: {processed_output}')

        except KeyboardInterrupt:
            print('Closing interactive mode.')
