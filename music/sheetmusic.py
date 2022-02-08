from automusicgen.data.dataset.music_token import MusicToken

from .conversion import to_stream


def show_sheet_music(music_tokens: list[MusicToken]):
    """
    Displays the music tokens as sheet music
    """
    music_stream = to_stream(music_tokens)
    music_stream.show()
