import random
from automusicgen.util.constants import SongPartitionMethod
from automusicgen.util.substring_util import get_all_substrings, get_all_substrings_of_size


def pass_through(song: str) -> list[str]:
    return [song]


def all_song_partitions(song: str) -> list[str]:
    partitions = get_all_substrings(song)
    return partitions

def _n_random_partitions(song: str, n: int) -> list[str]:
    song_partitions = all_song_partitions(song)
    randomly_selected_partitions = random.sample(song_partitions, n)
    return randomly_selected_partitions

def random_partitions_500(song: str) -> list[str]:
    return _n_random_partitions(song, 500)

def random_partitions(song: str, n: int, partition_size_limit = None):
    partitions = []
    for partition_idx in range(n):
        max_partition_size = min(n, max_partition_size) if partition_size_limit else n
        random_partition_size = random.randint(1, max_partition_size)

        max_start_idx = n - random_partition_size
        random_start_idx = random.randint(0, max_start_idx)

        end_idx = random_start_idx + random_partition_size
        
        partition = song[random_start_idx:end_idx]
        partitions.append(partition)

    return partitions

def random_partitions_1000(song: str) -> list[str]:
    return _n_random_partitions(song, 1000)

def _substrings_length_n(song: str, n: int) -> list[str]:
    return get_all_substrings_of_size(song, n)

SONG_PARTITION_METHOD_MAP = {
    SongPartitionMethod.NoPartition: pass_through,
    SongPartitionMethod.AllSubstrings: all_song_partitions,
    SongPartitionMethod.RandomSubstrings500: lambda song: _n_random_partitions(song, 500),
    SongPartitionMethod.RandomSubstrings1000: lambda song: _n_random_partitions(song, 1000),
    SongPartitionMethod.RandomSubstrings10000: lambda song: _n_random_partitions(song, 10000),
    SongPartitionMethod.RandomSubstrings25000: lambda song: _n_random_partitions(song, 25000),
    SongPartitionMethod.AllSubstringsLength50: lambda song: _substrings_length_n(song, 50),
    SongPartitionMethod.AllSubstringsLength70: lambda song: _substrings_length_n(song, 70),
}
