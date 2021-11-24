import torch
from config import Config


def get_device():
    """
    Retrieves the default device type. `cuda` is used if it is available, otherwise the
    `cpu` is used.

    Returns:
        str: The device type
    """
    if Config.args.cpu:
        return 'cpu'

    return 'cuda' if torch.cuda.is_available() else 'cpu'
