from __future__ import annotations

from enum import Enum, auto

from automusicgen.util.device import get_device
from torch import IntTensor


class MusicToken(Enum):
    Unknown = 0
    Pad = auto()
    BeginningOfSequence = auto()
    EndOfSequence = auto()
    A = auto()
    B = auto()
    C = auto()
    D = auto()
    E = auto()
    F = auto()
    G = auto()

    def __repr__(self) -> str:
        return _token_str_map[self] or f'<MusicToken.{self.name}>'

    def __str__(self) -> str:
        return _token_str_map[self] or f'<MusicToken.{self.name}>'

    @staticmethod
    def from_character(character: str) -> MusicToken:
        if character.upper() in {'A', 'B', 'C', 'D', 'E', 'F', 'G'}:
            return MusicToken[character.upper()]

        if character.upper() in {'X'}:
            return MusicToken.Pad

        if character == '<':
            return MusicToken.BeginningOfSequence
        
        if character == '>':
            return MusicToken.EndOfSequence

        return MusicToken.Unknown

    @staticmethod
    def to_string_list(music_tokens: list[MusicToken]) -> list[str]:
        return [str(music_token) for music_token in music_tokens]

    @staticmethod
    def to_joined_string(music_tokens: list[MusicToken], join_string: str = '') -> str:
        return join_string.join(MusicToken.to_string_list(music_tokens))

    @staticmethod
    def from_string(string: str) -> list[MusicToken]:
        return list(map(MusicToken.from_character, string))

    @staticmethod
    def as_values(music_tokens: list[MusicToken]) -> list[int]:
        return [music_token.value for music_token in music_tokens]

    @staticmethod
    def to_tensor(music_tokens: list[MusicToken]) -> IntTensor:
        return IntTensor(MusicToken.as_values(music_tokens)).to(device=get_device())

    @staticmethod
    def get_pad_token_value() -> int:
        return MusicToken.Pad.value

_token_str_map = {
    MusicToken.Unknown: '<unk>',
    MusicToken.Pad: '<pad>',
    MusicToken.BeginningOfSequence: '<bos>',
    MusicToken.EndOfSequence: '<eos>',
    MusicToken.A: 'a',
    MusicToken.B: 'b',
    MusicToken.C: 'c',
    MusicToken.D: 'd',
    MusicToken.E: 'e',
    MusicToken.F: 'f',
    MusicToken.G: 'g',
}
