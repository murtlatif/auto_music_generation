from enum import Enum
from string import punctuation

import torch

from automusicgen.config import Config
from automusicgen.data.dataset.music_token import MusicToken
from automusicgen.data.tokenize.midi_tokenizer import (MIDI_EOS_TOKEN,
                                                       MIDI_SOS_TOKEN, tokenizer)
from automusicgen.factory import get_model_with_parameters
from automusicgen.io.model_file_manager import (load_model,
                                                load_model_parameters)
from automusicgen.music.midi import show_midi
from automusicgen.music.sheetmusic import show_sheet_music
from automusicgen.parameter_search.testing_parameters import \
    TransformerParameters
from automusicgen.train.evaluator import generate_music, generate_music_midi
from automusicgen.util.device import get_device

from music.conversion import to_stream, to_stream_from_file


def config_setup():
    print('Running main.py with arguments:', Config.args)
    print(f'Device: {get_device()}')
    torch.manual_seed(Config.args.seed)

def validate_args():
    assert Config.args.load_param_path, "You must load model parameters into the playground."
    assert Config.args.load_model_path, "You must load a model into the playground."

def display_loaded_model_message(parameters: TransformerParameters):
    print(f'Loaded model with parameters: {parameters}')

def display_playground_message():
    print('-'*30)
    print('Welcome to the interactive playground!')
    print('Enter an input sequence and the model will generate music based on it.')
    print('Type "listen" before the input sequence to listen to the output.')
    print('Type "sheetmusic" before the input sequence to see the sheet music of the output.')
    print('Type "help" to see this message again.')
    print('Type "exit" to exit.')

def display_closing_message():
    print('Closing interactive playground.')

def remove_command_from_input(input_str: str) -> str:
    return ' '.join(input_str.split(' ')[1:])


if __name__ == '__main__':

    validate_args()

    load_param_path = Config.args.load_param_path
    parameters = load_model_parameters(load_param_path)

    model = get_model_with_parameters(parameters)

    # Load model state from file if state is given
    load_model_path = Config.args.load_model_path
    load_model(load_model_path, model)

    display_loaded_model_message(parameters)
    display_playground_message()
    try:
        while True:
            display_midi = False
            display_sheetmusic = False

            input_sequence = input('$ ').strip()
            
            # Process input commands
            command = input_sequence.lower().split(' ')[0]

            if command == 'exit':
                display_closing_message()
                break

            if command == 'help':
                display_playground_message()
                continue

            if command == 'listen':
                display_midi = True
                input_sequence = remove_command_from_input(input_sequence)

            if command == 'sheetmusic':
                display_sheetmusic = True
                input_sequence = remove_command_from_input(input_sequence)

            input_sequence_tokens = [int(token_str.strip(punctuation)) for token_str in input_sequence.split(' ')]

            # output_music = generate_music(model, input_sequence)
            output_music = generate_music_midi(model, input_sequence_tokens, sos_token=MIDI_SOS_TOKEN, eos_token=MIDI_EOS_TOKEN)
            output_music_midi = tokenizer.tokens_to_midi([output_music])

            tmp_midi_file = 'tmp_song.midi'
            output_music_midi.dump(tmp_midi_file)
            
            # output_music_string = MusicToken.to_joined_string(output_music, join_string=' ')
            output_midi_stream = to_stream_from_file(tmp_midi_file)

            output_midi_stream.show('text')

            if display_midi:
                print('Displaying MIDI...')
                output_midi_stream.show('midi')

            if display_sheetmusic:
                print('Displaying sheet music...')
                output_midi_stream.show()

    except KeyboardInterrupt:
        display_closing_message()
