from music21 import converter, stream

from ..data.dataset.music_token import MusicToken


def to_stream(music_tokens: list[MusicToken]) -> stream.Part:
    """
    Converts a list of music tokens into a stream of music
    """
    stringified_tokens = ' '.join(MusicToken.to_string_list(music_tokens))
    return converter.parse("tinynotation 4/4 " + stringified_tokens)


def show_sheet_music(music_tokens: list[MusicToken]):
    """
    Displays the music tokens as sheet music
    """
    music_stream = to_stream(music_tokens)
    music_stream.show()
