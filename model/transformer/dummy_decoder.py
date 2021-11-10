from torch import nn


class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask):
        return memory
