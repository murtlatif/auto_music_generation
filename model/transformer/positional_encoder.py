import math

import torch
from torch import nn


class PositionalEncoder(nn.Module):
    """
    Positional Encoder used for a transformer model.
    On the forward pass, attaches the encoded position of the input to
    the input itself and applies dropout.

    The encoding is generated up to a certain input length, and only
    the first N elements are taken where N is the length of the input.
    Each element is an array of size (hidden_dim,), so each input
    element is encoded with the number of hidden dimensions specified.

    Args:
        hidden_dim ([type]): The number of hidden dimensions to encode into each input element
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        max_len (int, optional): The maximum length of an input. Defaults to 100.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # The positional encoding is of shape (max_len, hidden_dim)
        positional_encoding = torch.zeros(max_len, hidden_dim)

        # Get the raw positions for the length dimension (max_len, 1)
        length_pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Get every other raw position for the hidden dimension (hidden_dim/2,)
        hidden_pos = torch.arange(0, hidden_dim, 2).float()
        hidden_encoding_values = -math.log(10000.0) / hidden_dim

        # Get the encoding for the hidden position (hidden_dim/2,)
        hidden_pos_encoding = torch.exp(hidden_pos * hidden_encoding_values)

        # Combine the length and hidden positions (max_len, hidden_dim/2)
        combined_position = length_pos * hidden_pos_encoding

        # Encode the position using sin as even positions, cos as odd positions
        positional_encoding[:, 0::2] = torch.sin(combined_position)
        positional_encoding[:, 1::2] = torch.cos(combined_position)

        # Transform encoding. Shape: (max_len, 1, hidden_dim)
        positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)

        # Store the positional encoding as part of the model's state
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        # Get the positional encoding up to the length of the input
        input_positionsal_encoding = self.positional_encoding[:x.size(0), :]

        # Attach the positional encoding to the input
        x = x + input_positionsal_encoding

        # Apply dropout
        return self.dropout(x)
