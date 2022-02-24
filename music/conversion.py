from automusicgen.data.dataset.music_token import MusicToken
from music21 import converter, stream


def to_stream(music_tokens: list[MusicToken]) -> stream.Part:
    """
    Converts a list of music tokens into a stream of music
    """
    music_tiny_notation = music_token_list_to_tiny_notation(music_tokens)

    return converter.parse(f'tinynotation: 4/4 {music_tiny_notation}')


def to_stream_from_file(midi_file: str) -> stream.Part:
    return converter.parse(midi_file)

def music_token_list_to_tiny_notation(music_tokens: list[MusicToken]) -> str:
    return ' '.join([
        music_token_to_tiny_notation(music_token) 
        for music_token in music_tokens 
        if music_token not in {MusicToken.BeginningOfSequence, MusicToken.EndOfSequence}
    ])

def music_token_to_tiny_notation(music_token: MusicToken) -> str:
    if music_token == MusicToken.Unknown or music_token == MusicToken.Pad:
        return 'r'

    if music_token == MusicToken.BeginningOfSequence or music_token == MusicToken.EndOfSequence:
        return ''

    return music_token.name.lower()
