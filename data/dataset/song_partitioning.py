import random
from automusicgen.util.constants import SongPartitionMethod
from automusicgen.util.substring_util import get_all_substrings


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

def random_partitions_1000(song: str) -> list[str]:
    return _n_random_partitions(song, 25000)

SONG_PARTITION_METHOD_MAP = {
    SongPartitionMethod.NoPartition: pass_through,
    SongPartitionMethod.AllSubstrings: all_song_partitions,
    SongPartitionMethod.RandomSubstrings500: random_partitions_500,
    SongPartitionMethod.RandomSubstrings1000: random_partitions_1000,
}
