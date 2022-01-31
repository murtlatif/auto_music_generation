import unittest
from model_main import get_all_song_windows


class TestDataMethods(unittest.TestCase):
    SAMPLE_SONG = "ABCDEFG"

    def test_get_all_song_windows(self):
        song_windows = get_all_song_windows(self.SAMPLE_SONG)
        correct_song_windows = ["AB", "ABC", "ABCD", "ABCDE", "ABCDEF", "ABCDEFG"]
        self.assertEqual(song_windows, correct_song_windows)

    def test_get_all_song_windows_empty(self):
        song_windows = get_all_song_windows("")
        correct_song_windows = []
        self.assertEqual(song_windows, correct_song_windows)


if __name__ == '__main__':
    unittest.main()
