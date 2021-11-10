import math

import torch
from torch import nn, Tensor
from util.constants import MUSIC_NOTES_LIST, PAD_TOKEN

from util.device import get_device

from .positional_encoder import PositionalEncoder
from .dummy_decoder import DummyDecoder


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
        self.positional_encoder = PositionalEncoder(hidden_dim, dropout=dropout)

        dummy_decoder = DummyDecoder()

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=feedforward_hidden_dim,
            dropout=dropout,
            custom_decoder=dummy_decoder,
        )

        # Fully connect and softmax the output
        self.fc_out = nn.Linear(hidden_dim, output_dict_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source: Tensor):
        """
        Takes in a batched input of source and target sequences and outputs
        another sequence.

        Args:
            source: Input sequence into the encoder, with shape (N, S, E)

            Where N is the batch size, S is the length of the source sequence,
            and E is the number of features.

        Returns:
            The output sequence
        """

        # Create a mask for the target
        mask = self.transformer.generate_square_subsequent_mask(source.shape[1]).to(get_device())

        # Embedded sequence has shape (N, S, E). Permute to (S, N, E).
        embedded_sequence = self.embedding(source).permute(1, 0, 2)

        # Encode the position into the sequence
        encoded_sequence = self.positional_encoder(embedded_sequence)

        # Get the output sequence
        embedded_output = self.transformer(src=encoded_sequence, tgt=encoded_sequence, src_mask=mask)

        # Repermute the output back to (N, S, E)
        embedded_output = embedded_output.permute(1, 0, 2)

        output = self.fc_out(embedded_output)
        return output

    @staticmethod
    def process_output(output: Tensor):
        """
        Processes the output tensor that this transformer produces into an
        array of notes.

        Args:
            output (Tensor): The output tensor to process

        Returns:
            list[list[str]]: The processed outputs
        """
        # Take the argmax across the features. Result is (N, S)
        argmaxed_output = output.argmax(axis=-1)

        processed_output = []

        for sequence in argmaxed_output:
            processed_sequence = [MUSIC_NOTES_LIST[note_idx] for note_idx in sequence]
            processed_output.append(processed_sequence)

        return processed_output
