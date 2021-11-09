import torch
from torch.utils.data import Dataset
from util.constants import PAD_TOKEN, note_map
from util.device import get_device


class MusicDataset(Dataset):
    def __init__(self, songs: list[str], note_vocab: str = "ABCDEFG", max_len_sequence: int = 32):

        self.songs = songs
        self.max_len_sequence = max_len_sequence

    def __getitem__(self, index: int):

        song = self.songs[index]

        song_len = len(song)
        
        self.x = torch.full((max_len_sequence,), PAD_TOKEN, device=get_device())
        self.tgt = torch.full((max_len_sequence,), PAD_TOKEN, device=get_device())


        return LongTensor(self.x[index]), LongTensor([0] + self.x[index])

    def __len__(self):
        return len(self.songs)

    @staticmethod
    def encode_notes(notes: str):
        """
        Converts a string of notes into a list of numbers corresponding to the
        note, where 1 is A and 7 is G. If the element is not found.

        Args:
            notes (str): A string containing the notes A-G.

        Raises:
            ValueError: If one of the strings in the note is not a valid note

        Returns:
            list[int]: A list containing the elements 1-7.
        """
        note_indices = [note_map.index(note) + 1 for note in notes]
        return note_indices
