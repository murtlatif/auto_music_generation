import torch
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from config import Config
from data.dataset.music_dataset import MusicDataset
from data.dataset.music_token import MusicToken
from model.transformer.transformer_baseline import TransformerModel
from train.evaluator import evaluate_note_sequence
from train.trainer import train_transformer
from util.constants import Songs
from util.device import get_device
from util.model_file_manager import load_model
from util.substring_util import get_random_substrings
from visualize.plot_model_stats import plot_accuracy, plot_loss

"""
Sample usage - Training a model:
    python model_main.py -vt --name v01 -s -soa --seed 29 -e 15

Sample usage - Evaluating a model:
    python model_main.py --seed 29 -i -l model/cache/tfmr_NAME_cpu_acc_1.00pt
"""

def initialize_config():
    print('Running main.py with arguments:', Config.args)
    print(f'Device: {get_device()}')
    torch.manual_seed(Config.args.seed)


def get_transformer_model(hidden_dim: int = 512, num_layers: int = 6) -> TransformerModel:
    """
    Creates the TransformerModel with an input/output dictionary size according
    to the number of tokens in MusicToken.

    Args:
        hidden_dim (int, optional): Number of features in the model. Defaults to 512.
        num_layers (int, optional): Number of layers in the model. Defaults to 6.

    Returns:
        TransformerModel: Transformer model with given parameters
    """
    dict_size = len(MusicToken)
    transformer_model = TransformerModel(
        input_dict_size=dict_size, hidden_dim=hidden_dim, num_layers=num_layers)

    return transformer_model


def get_music_data_loader(songs: list[str], batch_size: int, max_sequence_len: 'int | None' = None) -> DataLoader:
    # Default max sequence length to the maximum song length
    if max_sequence_len is None:
        max_sequence_len = max([len(song) for song in songs])

    dataset = MusicDataset(songs, max_sequence_length=max_sequence_len)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader


def get_song_snippets(song: str, len_snippets: int, num_snippets: int) -> list[str]:
    snippets = get_random_substrings(song, len_snippets, num_snippets)
    return snippets


def train_transformer_on_notes(
    model: TransformerModel,
    epochs: int = 10,
    batch_size: int = 32,
    print_status: bool = True,
):
    # Initialize parameters
    song_sample_length = 17

    # Initialize dataset
    all_songs = []
    # all_songs.append(Songs.TEST_SONG)
    # all_songs.extend(get_song_snippets(Songs.JINGLE_BELLS,
    #                  len_snippets=song_sample_length, num_snippets=128))
    # all_songs.extend(get_song_snippets(Songs.ODE_TO_JOY,
    #                  len_snippets=song_sample_length, num_snippets=64))
    # all_songs.extend(get_song_snippets(Songs.TWINKLE_TWINKLE,
    #                  len_snippets=song_sample_length, num_snippets=32))
    # all_songs.extend(get_song_snippets(Songs.OLD_MCDONALD,
    #                  len_snippets=song_sample_length, num_snippets=32))
    # all_songs.extend(get_song_snippets(Songs.ROW_YOUR_BOAT,
    #                  len_snippets=song_sample_length, num_snippets=16))
    # all_songs.extend(get_song_snippets(Songs.HAPPY_BIRTHDAY,
    #                  len_snippets=song_sample_length, num_snippets=16))
    all_songs.append(Songs.TWINKLE_TWINKLE)

    train_loader = get_music_data_loader(
        all_songs, batch_size, max_sequence_len=len(Songs.TWINKLE_TWINKLE) - 1)
        # all_songs, batch_size, max_sequence_len=song_sample_length-1)

    # Initialize training objects
    optimizer = optim.Adam(model.parameters(), lr=Config.args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # TODO: Update these!
    train_losses, train_accuracies, best_model_file = train_transformer(
        model,
        train_loader,
        optimizer,
        criterion,
        epochs,
        print_status,
        save_mode=Config.args.save_mode,
        save_on_accuracy=Config.args.save_on_accuracy
    )

    if Config.args.display:
        plot_loss(train_losses)
        plot_accuracy(train_accuracies)

    return best_model_file


if __name__ == '__main__':

    initialize_config()
    
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

                stringified_output = [list(map(str, batch)) for batch in processed_output]
                print(f'Output: {stringified_output}')

        except KeyboardInterrupt:
            print('Closing interactive mode.')
