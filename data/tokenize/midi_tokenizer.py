from miditok import REMI
from miditoolkit import MidiFile

def create_tokenizer(pitch_range, beat_resolution, num_velocities, additional_tokens) -> REMI:
    return REMI(pitch_range=pitch_range, beat_res=beat_resolution, nb_velocities=num_velocities, additional_tokens=additional_tokens)

def load_midi_file(midi_file_path: str) -> MidiFile:
    return MidiFile(filename=midi_file_path)

# Our parameters
pitch_range = range(21, 109)
beat_res = {(0, 4): 8, (4, 12): 4}
nb_velocities = 32
additional_tokens = {'Chord': True, 'Rest': False, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                     'rest_range': (2, 4),  # (half, 8 beats)
                     'nb_tempos': 64,  # nb of tempo bins
                     'tempo_range': (70, 210)}  # (min, max)

# Creates the tokenizer and loads a MIDI
tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, sos_eos_tokens=True)

MIDI_PAD_TOKEN = tokenizer.vocab.event_to_token['PAD_None']
MIDI_SOS_TOKEN = tokenizer.vocab.event_to_token['SOS_None']
MIDI_EOS_TOKEN = tokenizer.vocab.event_to_token['EOS_None']
VOCAB_SIZE = len(tokenizer.vocab)

def parse_midi_tracks(midi_file_path: str):
    midi = MidiFile(midi_file_path)
    tracks = tokenizer.midi_to_tokens(midi)
    return tracks

def parse_first_track_midi(midi_file_path: str):
    tracks = parse_midi_tracks(midi_file_path)
    return tracks[0]