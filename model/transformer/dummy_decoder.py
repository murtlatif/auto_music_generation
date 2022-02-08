from torch import nn


class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        return memory
