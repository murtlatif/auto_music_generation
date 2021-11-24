import torch
from torch.utils.data import Dataset
from util.device import get_device

from .music_token import MusicToken


class MusicDataset(Dataset):
    songs: list[str]
    max_sequence_length: int

    def __init__(self, songs: list[str], max_sequence_length: int = 32):
        self.songs = songs
        self.max_sequence_length = max_sequence_length

    def __getitem__(self, index: int):
        song = self.songs[index]
        encoded_song = MusicToken.to_tensor(MusicToken.from_string(song))

        song_len = len(song)

        PAD_TOKEN = MusicToken.get_pad_token_value()

        x = torch.full((self.max_sequence_length,), PAD_TOKEN, device=get_device())
        target = torch.full((self.max_sequence_length,), PAD_TOKEN, device=get_device())

        # No song: return the empty padded sequence
        if song_len == 0:
            return x, target

        # Song is shorter or equal to max sequence: Fill in until the song ends
        if song_len <= self.max_sequence_length:
            x[:song_len] = encoded_song
            target[:song_len-1] = encoded_song[1:]

        # Song is longer than max sequence: Take as much as we can
        else:
            truncated_song_data = encoded_song[:self.max_sequence_length + 1]
            x = truncated_song_data[:self.max_sequence_length]
            target = truncated_song_data[1:self.max_sequence_length + 1]

        return x, target

    def __len__(self):
        return len(self.songs)
