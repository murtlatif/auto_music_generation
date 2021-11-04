from string import ascii_uppercase
from typing import Sequence
from enum import Enum


# class Note:
#     def __init__(self, note: str):
#         note = note.upper()
#         if note not in ascii_uppercase[:7]:
#             raise ValueError("Invalid note provided")

#         self.note = note

#     def __repr__(self):
#         return self.note

#     @staticmethod
#     def from_characters(note_characters: Sequence[str]):
#         notes = []

#         for note_character in note_characters:
#             notes.append(Note(note_character))

#         return notes
