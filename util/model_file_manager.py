import os.path

import torch
from automusicgen.config import Config
from torch import nn

from .constants import DotenvKeys


def load_model(filename: str, model: nn.Module):
    """
    Loads a model state file into a model.

    Args:
        filename (str): The file containing the model state
        model (nn.Module): The model to load the state into
    """
    load_path = filename

    if not os.path.isfile(load_path):
        default_dir = Config.env[DotenvKeys.MODEL_DEFAULT_DIR]
        load_path = f'{default_dir}/{load_path}'

    assert os.path.isfile(load_path), f'Failed to find path {load_path}'

    model_state_dict = torch.load(load_path)
    model.load_state_dict(model_state_dict)


def save_model(filename: str, model: nn.Module):
    """
    Saves the model's state into a file.

    Args:
        filename (str): The file to save the model state
        model (nn.Module): The model whose state will be saved
    """
    save_model_state(filename, model.state_dict())


def save_model_state(filename: str, model_state):
    """
    Saves the model's state_dict into a file.
    """
    save_path = filename
    if not os.path.dirname(save_path):
        default_dir = Config.env[DotenvKeys.MODEL_DEFAULT_DIR]
        save_path = f'{default_dir}/{save_path}'

    torch.save(model_state, save_path)
