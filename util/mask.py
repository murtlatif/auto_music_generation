import torch
from torch import Tensor


def generate_square_subsequent_mask(size: int):
    """
    Generates an upper triangular matrix of shape `(size, size)` where
    values are 0 on and below the diagonal, and `-inf` above it.

    Used as a masking for the target sequence of the Transformer model
    to prevent the usage of future target values.

    Args:
        size (int): The length/width of the square mask

    Returns:
        Tensor: The mask.
    """

    mask = torch.triu(torch.ones(size, size), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))

    return mask


def generate_padding_mask(sequence: Tensor):
    """
    Creates a key padding mask of shape `(N, S)` where `N` is the batch size
    and `S` is the length of the sequence. 
    The elements of the mask are `True` if their corresponding element in the
    sequence is `0`, and is `False` otherwise.

    This will ignore all `0` elements in the sequence if this mask is passed
    into the Transformer model's [src/tgt/memory]_key_padding_mask value in 
    the forward pass.

    Args:
        sequence (Tensor): The sequence to create a mask from

    Returns:
        Tensor: A BoolTensor mask with shape `(N, S)`.
    """
    return (sequence == 0).transpose(0, 1)
