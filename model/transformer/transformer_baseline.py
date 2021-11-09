import math

import torch
from torch import nn, Tensor
from util.constants import PAD_TOKEN

from util.device import get_device

from .positional_encoder import PositionalEncoder


class TransformerModel(nn.Module):
    """
    Transformer Model that uses a positional encoder. Embeds the input
    sequences using an Embedding module and also embeds the position into the
    input sequences using a PositionalEncoder.

    The encoder and decoder components use the same model because with music
    notes we are not performing a task like translation, where the grammar or
    structure of the input may differ from the output.

    Args:
        input_dict_size (int): Size of input embedding dictionary
        output_dict_size (int): Size of output embedding dictionary
        hidden_dim (int, optional): The number of expected features in the encoder/decoder inputs. Defaults to 512.
        feedforward_hidden_dim (int, optional): The number of features in the feedforward model of the transformer.
        Defaults to 2048.
        num_layers (int, optional): The number of sub-encoder-layers in the encoder/decoder. Defaults to 6.
        num_heads (int, optional): The number of heads in the multiheadattention models. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
    """

    def __init__(
        self,
        input_dict_size: int,
        output_dict_size: int,
        hidden_dim: int = 512,
        feedforward_hidden_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(input_dict_size, hidden_dim, padding_idx=PAD_TOKEN)
        self.pos_encoder = PositionalEncoder(hidden_dim, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=feedforward_hidden_dim,
            dropout=dropout
        )

        # Fully connect and softmax the output
        self.fc_out = nn.Linear(hidden_dim, output_dict_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source: Tensor):
        """
        Takes in a batched input of source and target sequences and outputs
        another sequence.

        Args:
            source: Input sequence into the encoder, with shape (S, N, E)
            target: Target sequence into the decoder, with shape (T, N, E)

            Where S is the length of the source sequence, N is the batch size,
            and E is the number of features.

        Returns:
            The output sequence
        """

        # Create a mask for the target
        target_mask = self.transformer.generate_square_subsequent_mask(source.shape[1]).to(get_device())

        embedded_sequence: Tensor = self.embedding(source)
        
        # Generate the padding masks to ignore the 0s in the source/target
        # source_padding_mask = generate_padding_mask(source)
        # target_padding_mask = generate_padding_mask(target)

        # print(
        #     f'Source ({source.shape}): {source}, Target ({target.shape}: {target}')

        # print(
        # f'SrcPaddingMask ({source_padding_mask.shape}): {source_padding_mask}, TgtPaddingMask ({target_padding_mask.shape}): {target_padding_mask}')

        # Encode the source and target sequences
        source = self.encoder(source)
        # print(f'SrcEncoded ({source.shape}): {source}')
        source = self.pos_encoder(source)
        # print(f'SrcEncodedPos ({source.shape}): {source}')

        target = self.encoder(target)
        # print(f'TgtEncoded ({target.shape}): {target}')
        target = self.pos_encoder(target)
        # print(f'TgtEncodedPos ({target.shape}): {target}')

        # print(
        #     f'SrcMask: {self.source_mask}, TgtMask: {self.target_mask}')

        # Generate the output sequence
        output = self.transformer(source,
                                  target,
                                  src_mask=self.source_mask,
                                  tgt_mask=self.target_mask,
                                  memory_mask=self.memory_mask,
                                  #   src_key_padding_mask=source_padding_mask,
                                  #   tgt_key_padding_mask=target_padding_mask,
                                  #   memory_key_padding_mask=source_padding_mask)
                                  )

        print('OUT (', output.shape, ')')

        return output

        output = self.fc_out(output)

        return output
