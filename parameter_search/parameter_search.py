import json
import os

from automusicgen.config import Config
# from automusicgen.data.dataset.music_token import MusicToken
from automusicgen.data.tokenize.midi_tokenizer import (MIDI_EOS_TOKEN,
                                                       MIDI_SOS_TOKEN,
                                                       tokenizer)
from automusicgen.factory import (get_criterion, get_dataloader,
                                  get_midi_criterion, get_midi_dataloader,
                                  get_model_with_parameters, get_optimizer)
from automusicgen.model.transformer.transformer_baseline import \
    TransformerModel
from automusicgen.music.midi import write_midi
from automusicgen.parameter_search.search_experiments import (
    SWEET_CHILD_O_MINE_EXPERIMENT, TETRIS_EXPERIMENT, TETRIS_FULL_EXPERIMENT, UNDER_THE_SEA_EXPERIMENT)
from automusicgen.parameter_search.testing_parameters import \
    TransformerParameters
from automusicgen.train.evaluator import generate_music, generate_music_midi
from automusicgen.train.trainer import train_transformer
from automusicgen.util.constants import Songs
from automusicgen.util.string_formatter import format_percentage
from automusicgen.visualize.plot_model_stats import plot_accuracy, plot_loss
from torch import optim
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader

# from automusicgen.parameter_search.search_experiments import (
    # CHAOS_EXPERIMENT, JACK_EXPERIMENT, ODE_TO_JOY_EXPERIMENT, TEST_EXPERIMENT)


def _train_model_with_parameters(
        model: TransformerModel, 
        train_loader: DataLoader,
        optimizer: optim.Adam,
        criterion: _Loss,
        parameters: TransformerParameters
    ):
    train_losses, train_accuracies, best_model_file, best_param_file = train_transformer(
        model,
        train_loader,
        optimizer,
        criterion,
        parameters,
    )

    return train_losses, train_accuracies, best_model_file, best_param_file

def _evaluate_model_with_parameters(model: TransformerModel, model_output_directory: str, parameters: TransformerParameters):
    
    output_music = generate_music(model, parameters.test_song)

    song_name = list(Songs.__dict__.keys())[list(Songs.__dict__.values()).index(parameters.test_song)]
    midi_output_file = f'{model_output_directory}/generate_{song_name}.midi'
    write_midi(output_music, midi_output_file)
    
    return output_music

def _evaluate_model_midi(model: TransformerModel, model_output_directory: str, parameters: TransformerParameters):

    output_music_tokens = generate_music_midi(model, parameters.test_song, MIDI_SOS_TOKEN, MIDI_EOS_TOKEN)

    song_name = list(Songs.__dict__.keys())[list(Songs.__dict__.values()).index(parameters.test_song)]

    output_midi = tokenizer.tokens_to_midi([output_music_tokens])

    midi_output_file = f'{model_output_directory}/generate_{song_name}.midi'
    # write_midi(output_music, midi_output_file)
    output_midi.dump(midi_output_file)

    return midi_output_file

def search_parameters(parameter_list: list[TransformerParameters], experiment_name: str):
    model_results = []

    output_directory = f'parameter_search/out/{experiment_name}'

    for parameters in parameter_list:
        print(f'Experiment: {experiment_name} -> current model: {parameters.model_name}')

        if Config.args.verbose:
            print(f'[VERBOSE] With parameters: {parameters}')

        model_output_directory = f'{output_directory}/{parameters.model_name}'
        os.makedirs(model_output_directory, exist_ok=True)

        model = get_model_with_parameters(parameters)
        # train_loader = get_dataloader(parameters.training_songs, batch_size=Config.args.batch_size, song_partition_method=parameters.partition_method)
        train_loader = get_midi_dataloader(parameters.training_songs, batch_size=Config.args.batch_size, song_partition_method=parameters.partition_method)
        optimizer = get_optimizer(model, parameters)
        criterion = get_midi_criterion()

        train_losses, train_accuracies, best_model_file, best_param_file = _train_model_with_parameters(
            model,
            train_loader,
            optimizer,
            criterion,
            parameters
        )

        loss_plot_file = f'{model_output_directory}/loss_plot.jpg'
        accuracy_plot_file = f'{model_output_directory}/accuracy_plot.jpg'
        
        plot_loss(train_losses, False, loss_plot_file)
        plot_accuracy(train_accuracies, False, accuracy_plot_file)

        test_song_output_file = _evaluate_model_midi(model, model_output_directory, parameters)
        # test_song_output = _evaluate_model_with_parameters(model, model_output_directory, parameters)
        # test_song_output_string = MusicToken.to_joined_string(test_song_output, join_string=' ')

        results = {
            'Loss': f'[{min(train_losses):.4f}, {max(train_losses):.4f}]  Final: {train_losses[-1]:.4f}',
            'Accuracy': f'[{format_percentage(min(train_accuracies))}, {format_percentage(max(train_accuracies))}]  Final: {format_percentage(train_accuracies[-1])}',
            'Test Song Output': test_song_output_file,
            'Model File': best_model_file,
            'Parameters File': best_param_file,
            'Output Directory': model_output_directory,
        }

        with open(f'{model_output_directory}/results.json', 'w') as results_file:
            results_file.write(json.dumps(results, indent=2))

        model_results.append(results)

    return list(zip(parameter_list, model_results))

if __name__ == '__main__':
    # results = search_parameters(TEST_EXPERIMENT, 'TestExperiment')
    # results = search_parameters(GRAVITY_FALLS_ALL_PARTITIONS_EXPERIMENT, 'GravityFallsAllPartitionsExperiment')
    results = search_parameters(SWEET_CHILD_O_MINE_EXPERIMENT, 'SweetChildOMineExperiment')

    print('-' * 10, 'Testing Results', '-' * 10)
    for parameters, result in results:
        print(f'{parameters.model_name}: {json.dumps(result, indent=1)}')
    print('-' * 37)
