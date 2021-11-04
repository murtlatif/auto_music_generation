from torch.utils.data import Dataset

# from data.notes.note import Note
from sklearn.preprocessing import MultiLabelBinarizer

from torch import LongTensor


class SimpleNotes(Dataset):
    def __init__(self, notes, note_vocab: str = "ABCDEFG", len_sequence: int = 4):
        # self.multi_label_binarizer = MultiLabelBinarizer()
        # self.multi_label_binarizer.fit(note_vocab.upper())

        # self.notes = notes.upper()
        self.len_sequence = len_sequence

        self.x = [notes[i:i+len_sequence] for i in range(len(notes) - len_sequence + 1)]

        # self.samples = [self.encode(self.notes[i:i+len_sequence]) for i in range(len(notes) - len_sequence + 1)]
        # self.sample_targets = [self.encode([0] + (self.notes[i:i+len_sequence-1]))
        #                        for i in range(len(notes) - len_sequence + 1)]

    def __getitem__(self, index: int):
        return LongTensor(self.x[index]), LongTensor([0] + self.x[index])

    def __len__(self):
        return len(self.x)

    def encode(self, notes):
        return self.multi_label_binarizer.transform(notes)
