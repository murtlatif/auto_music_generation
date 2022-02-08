import unittest

from .substring_util import get_all_substrings


class TestSubstringMethods(unittest.TestCase):
    SAMPLE_SONG = "ABCD"

    def test_get_all_substrings(self):
        substrings = get_all_substrings(self.SAMPLE_SONG)
        correct_song_windows = ['A', 'AB', 'ABC', 'ABCD', 'B', 'BC', 'BCD', 'C', 'CD', 'D']
        self.assertEqual(substrings, correct_song_windows)

    def test_get_all_substrings_empty(self):
        substrings = get_all_substrings('')
        correct_song_windows = []
        self.assertEqual(substrings, correct_song_windows)

if __name__ == '__main__':
    unittest.main()

