import os.path

from config import Config
from constants import DotenvKeys


def get_song_file_path(song_file: str):
    song_dir = Config.env.fetch[DotenvKeys.MUSIC_DIR]
    return f'{song_dir}/{song_file}'


def get_and_verify_song_path_from_config():
    song_file = Config.args.song
    song_path = get_song_file_path(song_file)
    assert os.path.isfile(song_path), f'Song file "{song_path}" does not exist!'

    return song_path
