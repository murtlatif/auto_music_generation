from enum import Enum, auto

from automusicgen.data.tokenize.midi_tokenizer import parse_first_track_midi
from miditoolkit import MidiFile


class SaveMode(Enum):
    NoSave = 'no-save'
    SaveLast = 'save-last'
    SaveBest = 'save-best'
    SaveEachNewBest = 'save-all-best'

    def __str__(self):
        return self.value

class SongPartitionMethod(Enum):
    NoPartition = 'no-partition',
    AllSubstrings = 'all-substrings',
    RandomSubstrings500 = 'random-substrings-500',
    RandomSubstrings1000 = 'random-substrings-1000',

    def __str__(self):
        return self.value


class DotenvKeys:
    MODEL_DEFAULT_DIR = 'MODEL_DEFAULT_DIR'
    MUSIC_DIR = 'MUSIC_DIR'

class MidiFiles:
    GRAVITY_FALLS = 'assets/audio/music/gravity_falls_theme.mid'
    UNDER_THE_SEA = 'assets/audio/music/under_the_sea.mid'
    TETRIS = 'assets/audio/music/tetris_theme.mid'
    SWEET_CHILD_O_MINE = 'assets/audio/music/sweet_child_o_mine.mid'

class Songs:
    GRAVITY_FALLS = parse_first_track_midi(MidiFiles.GRAVITY_FALLS)
    UNDER_THE_SEA = parse_first_track_midi(MidiFiles.UNDER_THE_SEA)
    TETRIS = parse_first_track_midi(MidiFiles.TETRIS)
    SWEET_CHILD_O_MINE = parse_first_track_midi(MidiFiles.SWEET_CHILD_O_MINE)

class Songs2:
    TEST_SONG = "ABABABABABABABABABABABABABAB"
    OLD_MCDONALD = "CCCGAAGGEEDDCCCGCCCGAAGGEEDDCCCG"
    TWINKLE_TWINKLE = "CCGGAAGGFFEEDDCCGGFFEEDDGGFFEEDDCCGGAAGGFFEEDDCC"
    ODE_TO_JOY = "EEFGGFEDCCDEEDDDEEFGGFEDCCDEDCCCDDECDFECDFEDCDGGEEFGGFEDCCDEDCC"
    HAPPY_BIRTHDAY = "GGAGCBGGAGDCGGGECBAFFECDC"
    ROW_YOUR_BOAT = "CCCDEEDEFGCCCGGGEEECCCGFEDC"
    JINGLE_BELLS = "CAGFCCCCAGFDDDDDBAGEEEECCBGAAAACAGFCCCCCAGFDDDDDBAGCCCCDCBGFAAAAAAAAACFGABBBBBAAAAGGAGGCCAAAAAAAAACFGAAAABBBBBAAACCBGF"
