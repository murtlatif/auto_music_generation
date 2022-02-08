from automusicgen.data.dataset.music_token import MusicToken

from .conversion import to_stream


def show_midi(music_tokens: list[MusicToken]):
    """
    Converts and displays the music tokens as a MIDI music format.
    """
    music_stream = to_stream(music_tokens)
    music_stream.show('midi')

def write_midi(music_tokens: list[MusicToken], file_path: str):
    """
    Writes the MIDI file to the given file path.
    """
    music_stream = to_stream(music_tokens)
    music_stream.write('midi', fp=file_path)
