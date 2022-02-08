import unittest

from .song_partitioning import all_song_partitions, pass_through


class TestMusicDataset(unittest.TestCase):
    SAMPLE_SONG = 'ABC'

    def test_pass_through_partition(self):
        partitioned_song = pass_through(self.SAMPLE_SONG)

        TARGET = ['ABC']
        self.assertListEqual(partitioned_song, TARGET)


    def test_all_song_partitions(self):
        partitioned_song = all_song_partitions(self.SAMPLE_SONG)

        TARGET = ['A', 'AB', 'ABC', 'B', 'BC', 'C']
        self.assertListEqual(partitioned_song, TARGET)


if __name__ == '__main__':
    unittest.main()
