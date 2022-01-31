from __future__ import annotations

from enum import Enum, auto

from torch import LongTensor


class MusicToken(Enum):
    UNKNOWN = 0
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()

    def __repr__(self) -> str:
        return f'<MusicToken.{self.name}>'

    def __str__(self):
        if self == MusicToken.UNKNOWN:
            return '?'
        
        return self.name

    @staticmethod
    def from_character(character: str) -> MusicToken:
        if character.upper() in {'A', 'B', 'C', 'D', 'E', 'F', 'G'}:
            return MusicToken[character.upper()]

        return MusicToken.UNKNOWN

    @staticmethod
    def to_string_list(music_tokens: list[MusicToken]) -> list[str]:
        return [str(music_token) for music_token in music_tokens]

    @staticmethod
    def to_joined_string(music_tokens: list[MusicToken]) -> str:
        return ''.join(MusicToken.to_string_list(music_tokens))

    @staticmethod
    def from_string(string: str) -> list[MusicToken]:
        return list(map(MusicToken.from_character, string))

    @staticmethod
    def as_values(music_tokens: list[MusicToken]) -> list[int]:
        return [music_token.value for music_token in music_tokens]

    @staticmethod
    def to_tensor(music_tokens: list[MusicToken]) -> LongTensor:
        return LongTensor(MusicToken.as_values(music_tokens))

    @staticmethod
    def get_pad_token_value() -> int:
        return MusicToken.UNKNOWN.value
