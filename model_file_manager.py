import os.path

import torch

from config import Config
from constants import DotenvKeys


def load_model(filename: str, model: object):
    load_path = filename

    if not os.path.isfile(load_path):
        default_dir = Config.env.fetch(DotenvKeys.MODEL_DEFAULT_DIR)
        load_path = f'{default_dir}/{load_path}'

    assert os.path.isfile(load_path), f'Failed to find path {load_path}'

    model_state_dict = torch.load(load_path)
    model.load_state_dict(model_state_dict)


def save_model(filename: str, model: object):
    save_path = filename
    if os.path.dirname(save_path):
        default_dir = Config.env.fetch(DotenvKeys.MODEL_DEFAULT_DIR)
        save_path = f'{default_dir}/{save_path}'

    torch.save(model.state_dict(), save_path)
