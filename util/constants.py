from .substring_util import get_all_substrings


class DotenvKeys:
    MODEL_DEFAULT_DIR = 'MODEL_DEFAULT_DIR'
    MUSIC_DIR = 'MUSIC_DIR'


class Songs:
    TEST_SONG = "ABABABABABABABAB"
    OLD_MCDONALD = "CCCGAAGGEEDDCCCGCCCGAAGGEEDDCCCG"
    TWINKLE_TWINKLE = "CCGGAAGGFFEEDDCCGGFFEEDDGGFFEEDDCCGGAAGGFFEEDDCC"
    ODE_TO_JOY = "EEFGGFEDCCDEEDDDEEFGGFEDCCDEDCCCDDECDFECDFEDCDGGEEFGGFEDCCDEDCC"
    HAPPY_BIRTHDAY = "GGAGCBGGAGDCGGGECBAFFECDC"
    ROW_YOUR_BOAT = "CCCDEEDEFGCCCGGGEEECCCGFEDC"
    JINGLE_BELLS = "CAGFCCCCAGFDDDDDBAGEEEECCBGAAAACAGFCCCCCAGFDDDDDBAGCCCCDCBGFAAAAAAAAACFGABBBBBAAAAGGAGGCCAAAAAAAAACFGAAAABBBBBAAACCBGF"
    TEST_SONG = "ABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABABAB"


class SongSnippets:
    TEST_SONG = get_all_substrings(Songs.JINGLE_BELLS)
    ODE_TO_JOY = get_all_substrings(Songs.ODE_TO_JOY)
    TWINKLE_TWINKLE = get_all_substrings(Songs.TWINKLE_TWINKLE)
    OLD_MCDONALD = get_all_substrings(Songs.OLD_MCDONALD)
    ROW_YOUR_BOAT = get_all_substrings(Songs.ROW_YOUR_BOAT)
    HAPPY_BIRTHDAY = get_all_substrings(Songs.HAPPY_BIRTHDAY)
