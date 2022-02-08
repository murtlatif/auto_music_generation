import unittest

from automusicgen.util.constants import SongPartitionMethod

from .music_dataset import MusicDataset
from .music_token import MusicToken


class TestMusicDataset(unittest.TestCase):
    SAMPLE_SONG_1 = "AB"
    SAMPLE_SONG_2 = "ABCD"
    
    def test_dataset_length(self):
        songs = [self.SAMPLE_SONG_2]
        dataset = MusicDataset(songs)
        self.assertEqual(len(dataset), len(songs))


    def test_basic_data_sample(self):
        songs = [self.SAMPLE_SONG_2]
        dataset = MusicDataset(songs)
        source, target = dataset[0]

        source = source.tolist()
        target = target.tolist()

        CORRECT_SOURCE = MusicToken.to_tensor(MusicToken.from_string("<ABCD")).tolist()
        CORRECT_TARGET = MusicToken.to_tensor(MusicToken.from_string("<ABCD>")).tolist()

        self.assertListEqual(source, CORRECT_SOURCE)
        self.assertListEqual(target, CORRECT_TARGET)

    def test_variable_length_data_sample(self):
        songs = [self.SAMPLE_SONG_2]
        dataset = MusicDataset(songs, SongPartitionMethod.AllSubstrings)

        source, target = dataset[0]

        source = source.tolist()
        target = target.tolist()

        CORRECT_SOURCE = MusicToken.to_tensor(MusicToken.from_string("<XXXX")).tolist()
        CORRECT_TARGET = MusicToken.to_tensor(MusicToken.from_string("<AXXXX")).tolist()

        self.assertListEqual(source, CORRECT_SOURCE)
        self.assertListEqual(target, CORRECT_TARGET)

        source, target = dataset[1]

        source = source.tolist()
        target = target.tolist()

        CORRECT_SOURCE = MusicToken.to_tensor(MusicToken.from_string("<AXXX")).tolist()
        CORRECT_TARGET = MusicToken.to_tensor(MusicToken.from_string("<ABXXX")).tolist()

        self.assertListEqual(source, CORRECT_SOURCE)
        self.assertListEqual(target, CORRECT_TARGET)

    def test_partition_method_no_partition(self):
        songs = [self.SAMPLE_SONG_1]
        dataset = MusicDataset(songs, song_partition_method=SongPartitionMethod.NoPartition)

        CORRECT_PARTITION_STRINGS = ['<AB>']
        CORRECT_PARTITIONS = [
            MusicToken.as_values(MusicToken.from_string(partition_string)) 
            for partition_string in CORRECT_PARTITION_STRINGS
        ]
        dataset_partitions = [partition.tolist() for partition in dataset.partitioned_songs]

        self.assertEqual(len(dataset_partitions), 1)
        self.assertEqual(dataset_partitions, CORRECT_PARTITIONS)


    def test_partition_method_all_substrings(self):
        songs = [self.SAMPLE_SONG_1]
        dataset = MusicDataset(songs, song_partition_method=SongPartitionMethod.AllSubstrings)

        CORRECT_PARTITION_STRINGS = ['<A', '<AB', '<AB>', 'AB', 'AB>', 'B>']
        CORRECT_PARTITIONS = [
            MusicToken.as_values(MusicToken.from_string(partition_string)) 
            for partition_string in CORRECT_PARTITION_STRINGS
        ]
        dataset_partitions = [partition.tolist() for partition in dataset.partitioned_songs]

        self.assertEqual(len(dataset_partitions), 6)
        self.assertEqual(dataset_partitions, CORRECT_PARTITIONS)


if __name__ == '__main__':
    unittest.main()
