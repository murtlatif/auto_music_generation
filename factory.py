

from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

from automusicgen.data.dataset.midi_token_dataset import MidiTokenDataset
from automusicgen.data.dataset.music_dataset import MusicDataset
from automusicgen.data.dataset.music_token import MusicToken
from automusicgen.data.tokenize.midi_tokenizer import MIDI_PAD_TOKEN
from automusicgen.model.transformer.transformer_baseline import \
    TransformerModel
from automusicgen.parameter_search.testing_parameters import \
    TransformerParameters
from automusicgen.util.constants import SongPartitionMethod
from automusicgen.util.device import get_device


def get_model_with_parameters(parameters: TransformerParameters) -> TransformerModel:
    return TransformerModel(
        parameters.input_dict_size,
        parameters.output_dict_size,
        parameters.hidden_dim,
        parameters.feedforward_hidden_dim,
        parameters.num_layers,
        parameters.num_heads,
        parameters.dropout,
        device=get_device()
    )


def get_optimizer(model: TransformerModel, parameters: TransformerParameters) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(), lr=parameters.learning_rate)
    return optimizer

def get_criterion():
    return nn.CrossEntropyLoss(ignore_index=MusicToken.get_pad_token_value())

def get_midi_criterion():
    return nn.CrossEntropyLoss(ignore_index=MIDI_PAD_TOKEN)

def get_dataloader(songs: list[str], batch_size: int, song_partition_method: SongPartitionMethod = SongPartitionMethod.AllSubstrings):
    dataset = MusicDataset(songs, song_partition_method=song_partition_method)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader

def get_midi_dataloader(songs: list[list[int]], batch_size: int, song_partition_method: SongPartitionMethod = SongPartitionMethod.RandomSubstrings1000):
    dataset = MidiTokenDataset(songs, song_partition_method)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader
