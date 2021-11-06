import math

import torch
from torch import nn
from util.mask import generate_padding_mask, generate_square_subsequent_mask

from .positional_encoder import PositionalEncoder


class TransformerModel(nn.Module):
    """
    Transformer Model that uses a positional encoder. Encodes the
    input and decodes the output using an Embedding module.

    Args:
        input_dict_size (int): Size of input embedding dictionary
        output_dict_size (int): Size of output embedding dictionary
        hidden_dim (int): The number of expected features in the encoder/decoder inputs.
        num_layers (int, optional): The number of sub-encoder-layers in the encoder/decoder.
        Defaults to 3.
        num_heads (int, optional): The number of heads in the multiheadattention models.
        Defaults to 2.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
    """

    def __init__(self, input_dict_size: int, output_dict_size: int, hidden_dim: int, num_layers: int = 3, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()

        # Define the encoders / decoders
        self.encoder = nn.Embedding(input_dict_size, hidden_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoder(hidden_dim, dropout=dropout)

        self.decoder = nn.Embedding(
            output_dict_size, hidden_dim, padding_idx=0)
        self.pos_decoder = PositionalEncoder(hidden_dim, dropout)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )

        # Create the fully connected layer to map the embedded features
        # to the output values
        self.fc_out = nn.Linear(hidden_dim, output_dict_size)

        # Mask definitions
        self.source_mask = None
        self.target_mask = None
        self.memory_mask = None

    def forward(self, source, target):
        # Create a mask for the target
        if self.target_mask is None or self.target_mask.size(0) != len(target):
            self.target_mask = generate_square_subsequent_mask(
                len(target)).to(target.device)

        # Generate the padding masks to ignore the 0s in the source/target
        source_padding_mask = generate_padding_mask(source)
        target_padding_mask = generate_padding_mask(target)

        print(
            f'Source ({source.shape}): {source}, Target ({target.shape}: {target}')

        print(
            f'SrcPaddingMask ({source_padding_mask.shape}): {source_padding_mask}, TgtPaddingMask ({target_padding_mask.shape}): {target_padding_mask}')

        # Encode the source and target sequences
        source = self.encoder(source)
        print(f'SrcEncoded ({source.shape}): {source}')
        source = self.pos_encoder(source)
        print(f'SrcEncodedPos ({source.shape}): {source}')

        target = self.decoder(target)
        print(f'TgtEncoded ({target.shape}): {target}')
        target = self.pos_decoder(target)
        print(f'TgtEncodedPos ({target.shape}): {target}')

        print(
            f'SrcMask: {self.source_mask}, TgtMask: {self.target_mask}')

        # Generate the output sequence
        output = self.transformer(source,
                                  target,
                                  src_mask=self.source_mask,
                                  tgt_mask=self.target_mask,
                                  memory_mask=self.memory_mask,
                                  src_key_padding_mask=source_padding_mask,
                                  tgt_key_padding_mask=target_padding_mask,
                                  memory_key_padding_mask=source_padding_mask)

        output = self.fc_out(output)

        return output
