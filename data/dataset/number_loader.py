from typing import Sequence
from torch import LongTensor
from torch.utils.data import Dataset


class NumberLoader(Dataset):
    def __init__(self, x: Sequence[int], y: Sequence[int], input_len: int = 3, output_len: int = 3):
        assert len(x) == len(y), "len(x) must equal len(y)"

        self.x = [[x[i + j] for j in range(input_len)] for i in range(len(x) - input_len + 1)]
        self.y = [[y[i + j] for j in range(output_len)] for i in range(len(y) - output_len + 1)]

    def __getitem__(self, index: int):
        return LongTensor(self.x[index]), LongTensor([0] + self.y[index])

    def __len__(self):
        return len(self.x)
