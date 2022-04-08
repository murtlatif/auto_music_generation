from typing import Callable

import torch
from automusicgen.config import Config
from automusicgen.data.tokenize.midi_tokenizer import MIDI_PAD_TOKEN
from automusicgen.util.constants import SongPartitionMethod
from automusicgen.util.device import get_device
from torch.utils.data import Dataset

from .song_partitioning import SONG_PARTITION_METHOD_MAP


class MidiTokenDataset(Dataset):

    def __init__(self, songs: list[int], song_partition_method: SongPartitionMethod = SongPartitionMethod.RandomSubstrings1000):
        self.songs = songs
        self.preprocessed_songs = self._preprocess_songs(songs)
        # self.max_sequence_length = max([len(song) for song in self.preprocessed_songs]) - 1
        self.max_sequence_length = Config.args.max_len

        self.song_partition_method = song_partition_method
        self.song_partitioner = SONG_PARTITION_METHOD_MAP[song_partition_method]
        self.partitioned_songs = self._partition_songs(self.song_partitioner, self.preprocessed_songs)

        if Config.args.verbose > 0:
            print(f'Initialized dataset with {len(songs)} songs. Partitioned into {len(self.partitioned_songs)} training samples, with max sequence length: {self.max_sequence_length}')

    def __getitem__(self, index: int):
        """
        A partitioned song has N tokens:
            <ABCD>  (N = 6)

        The first [:N-1] notes are the source 
            <ABCD   (size N-1)
        The last [:N] notes are the target 
            <ABCD>  (size N)

        target[:-1] is used as input to a model, and target[1:] is used as output

        self.max_sequence_length is equal to N
        """

        partitioned_song = self.partitioned_songs[index]
        partitioned_song_len = len(partitioned_song)

        PAD_TOKEN = MIDI_PAD_TOKEN

        x = torch.full((self.max_sequence_length,), PAD_TOKEN, device=get_device())
        target = torch.full((self.max_sequence_length+1,), PAD_TOKEN, device=get_device())

        # No song: return the empty padded sequence
        if partitioned_song_len == 0:
            return x, target

        # Song is shorter or equal to max sequence: Fill in until the song ends
        if partitioned_song_len <= self.max_sequence_length:
            x[:partitioned_song_len-1] = partitioned_song[:-1]
            target[:partitioned_song_len] = partitioned_song[:]

        # Song is longer than max sequence: Take as much as we can
        else:
            truncated_song_data = partitioned_song[:self.max_sequence_length + 1]
            x = truncated_song_data[:self.max_sequence_length]
            target = truncated_song_data[:self.max_sequence_length + 1]

        return x, target


    def __len__(self):
        return len(self.partitioned_songs)
    

    def _preprocess_songs(self, songs):
        return [torch.LongTensor(song).to(device=get_device()) for song in songs]

    def _partition_songs(self, partitioner: Callable[[str], list[str]], songs: list[int]) -> list[int]:
        partitioned_songs = []
        for song in songs:
            partitioned_songs.extend(partitioner(song))

        return [partitioned_song for partitioned_song in partitioned_songs if len(partitioned_song) > 1]

