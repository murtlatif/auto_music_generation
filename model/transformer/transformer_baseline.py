
import torch
from automusicgen.data.dataset.music_token import MusicToken
from automusicgen.data.tokenize.midi_tokenizer import MIDI_PAD_TOKEN
from automusicgen.util.device import get_device
from torch import Tensor, nn

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
        output_dict_size (int): Size of output embedding dictionary. Defaults to the input_dict_size.
        hidden_dim (int, optional): The number of expected features in the encoder/decoder inputs. Defaults to 512.
        feedforward_hidden_dim (int, optional): The number of features in the feedforward model of the transformer.
        Defaults to 2048.
        num_layers (int, optional): The number of sub-encoder-layers in the encoder/decoder. Defaults to 6.
        num_heads (int, optional): The number of heads in the multiheadattention models. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        device (str, optional): The device to use. Defaults to 'cuda' if available, otherwise 'cpu'.
    """

    def __init__(
        self,
        input_dict_size: int,
        output_dict_size: int = None,
        hidden_dim: int = 512,
        feedforward_hidden_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: str = get_device(),
    ):
        super().__init__()

        self.PAD_TOKEN = MIDI_PAD_TOKEN
        self.DEVICE = device

        if output_dict_size is None:
            output_dict_size = input_dict_size

        self.embedding = nn.Embedding(input_dict_size, hidden_dim, padding_idx=self.PAD_TOKEN, device=device)
        self.positional_encoder = PositionalEncoder(hidden_dim, dropout=dropout, device=device)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=feedforward_hidden_dim,
            dropout=dropout,
            device=device,
        )

        # Fully connect and softmax the output
        self.fc_out = nn.Linear(hidden_dim, output_dict_size, device=device)

    def forward(
        self,
        source: Tensor,
        target: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
        source_padding_mask: Tensor,
        target_padding_mask: Tensor,
    ):
        """
        Takes in a batched input of source and target sequences and outputs
        another sequence.

        Args:
            source: Input source sequence into the encoder, with shape (N, S, E)
            target: Input target sequence into the encoder, with shape (N, T, E)
            source_mask: Masking for the source sequence. Generally use zeros. Shape (S, S)
            target_mask: Masking for the target sequence. Generally use a matrix where the diagonal
                and lower triangle are 0, and the remaining values are -inf. Shape (T, T)
            source_padding_mask: A mask of True/False where True indicates an index that should be
                considered as padding, and will be skipped over. Shape (N, S)
            target_padding_mask: A mask of True/False where True indicates an index that should be
                considered as padding, and will be skipped over. Shape (N, T)
                
            Where N is the batch size, S is the length of the source sequence,
            T is the length of the target sequence, and E is the number of features (hidden dim).

        Returns:
            Tensor: The output sequence
        """

        source = source.to(device=self.DEVICE)
        target = target.to(device=self.DEVICE)

        # Embedded sequence has shape (N, S, E). Permute to (S, N, E).
        embedded_source_sequence = self.embedding(source).permute(1, 0, 2)
        embedded_target_sequence = self.embedding(target).permute(1, 0, 2)

        # Encode the position into the sequence
        encoded_source_sequence = self.positional_encoder(embedded_source_sequence)
        encoded_target_sequence = self.positional_encoder(embedded_target_sequence)

        # Get the output sequence and permute from (S, N, E) to (N, S, E)
        embedded_output = self.transformer(
            src=encoded_source_sequence,
            tgt=encoded_target_sequence,
            src_mask=source_mask,
            tgt_mask=target_mask,
            src_key_padding_mask=source_padding_mask,
            tgt_key_padding_mask=target_padding_mask
        )
        embedded_output = embedded_output.permute(1, 0, 2)

        # Output is of shape (N, S, Vout) where Vout is the size of the dictionary (MusicToken)
        output = self.fc_out(embedded_output)
        return output

    def encode(self, source: Tensor, source_mask: Tensor):
        """
        Encodes the source tensor.

        Shape of source tensor is (S, N)
        Shape of embedded tensor is (S, N, E)

        where S is the length of the sequence, N is the batch size, and E is the number of features.
        """
        embedded_source_sequence = self.embedding(source)
        encoded_source_sequence = self.positional_encoder(embedded_source_sequence)
        return self.transformer.encoder(encoded_source_sequence, source_mask)


    def decode(self, target: Tensor, memory: Tensor, target_mask: Tensor):
        embedded_target_sequence = self.embedding(target)
        encoded_target_sequence = self.positional_encoder(embedded_target_sequence)

        return self.transformer.decoder(encoded_target_sequence, memory, tgt_mask=target_mask)


    def create_mask(self, source, target):
        src_seq_len = source.shape[1]
        tgt_seq_len = target.shape[1]

        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.DEVICE).type(torch.bool)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(device=self.DEVICE)

        src_padding_mask = (source == self.PAD_TOKEN)
        tgt_padding_mask = (target == self.PAD_TOKEN)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    @staticmethod
    def process_output(output: Tensor) -> list[list[MusicToken]]:
        """
        Processes the output tensor that this transformer produces into a list
        of MusicToken objects.

        Args:
            output (Tensor): The output tensor to process

        Returns:
            list[list[MusicToken]]: The processed outputs
        """
        # Take the argmax across the features. Result is (N, S)
        argmaxed_output = output.argmax(axis=-1)

        processed_output: list[list[MusicToken]] = []

        for sequence in argmaxed_output:
            processed_sequence = [MusicToken(
                note_idx.item()) for note_idx in sequence]
            processed_output.append(processed_sequence)

        return processed_output
