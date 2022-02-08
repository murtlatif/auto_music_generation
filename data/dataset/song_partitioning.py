from automusicgen.util.constants import SongPartitionMethod
from automusicgen.util.substring_util import get_all_substrings


def pass_through(song: str) -> list[str]:
    return [song]


def all_song_partitions(song: str) -> list[str]:
    partitions = get_all_substrings(song)
    return partitions

SONG_PARTITION_METHOD_MAP = {
    SongPartitionMethod.NoPartition: pass_through,
    SongPartitionMethod.AllSubstrings: all_song_partitions,
}
