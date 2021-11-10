import torch
from torch import LongTensor
from torch.utils.data import Dataset
from util.constants import MUSIC_NOTES_TO_INDEX_MAP, PAD_TOKEN, MUSIC_NOTES_LIST
from util.device import get_device


class MusicDataset(Dataset):
    def __init__(self, songs: list[str], max_sequence_length: int = 32):

        self.songs = songs
        self.max_sequence_length = max_sequence_length

    def __getitem__(self, index: int):

        song = self.songs[index]
        encoded_song = MusicDataset.encode_notes(song)

        song_len = len(song)

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

    @staticmethod
    def encode_notes(notes: str) -> LongTensor:
        """
        Converts a string of notes into a list of numbers corresponding to the
        note, where the range 1-7 corresponds to the range A-G. The character
        "?" is mapped to 0. If the element is not found, a ValueError is raised.

        Args:
            notes (str): A string containing the notes [?A-G]

        Raises:
            ValueError: If one of the strings in the note is not a valid note

        Returns:
            list[int]: A list containing the elements [0-7]
        """
        note_indices = torch.LongTensor([MUSIC_NOTES_TO_INDEX_MAP[note.upper()] for note in notes])
        return note_indices
