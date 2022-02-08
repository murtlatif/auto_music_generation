import json
import os

from automusicgen.config import Config
from automusicgen.data.dataset.music_dataset import MusicDataset
from automusicgen.data.dataset.music_token import MusicToken
from automusicgen.model.transformer.transformer_baseline import \
    TransformerModel
from automusicgen.music.midi import write_midi
from automusicgen.parameter_search.search_experiments import TEST_EXPERIMENT, TWINKLE_TWINKLE_EXPERIMENT
from automusicgen.parameter_search.testing_parameters import TestingParameters
from automusicgen.train.evaluator import generate_music
from automusicgen.train.trainer import train_transformer
from automusicgen.util.constants import SaveMode, SongPartitionMethod, Songs
from automusicgen.util.device import get_device
from automusicgen.util.string_formatter import format_percentage
from automusicgen.visualize.plot_model_stats import plot_accuracy, plot_loss
from torch import nn, no_grad, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader


def _get_model_with_parameters(parameters: TestingParameters) -> TransformerModel:
    model = TransformerModel(
        parameters.input_dict_size,
        parameters.output_dict_size,
        parameters.hidden_dim,
        parameters.feedforward_hidden_dim,
        parameters.num_layers,
        parameters.num_heads,
        parameters.dropout,
        device=get_device()
    )

    return model

def _get_optimizer(model: TransformerModel, parameters: TestingParameters) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(), lr=parameters.learning_rate)
    return optimizer

def _get_criterion():
    return nn.CrossEntropyLoss(ignore_index=MusicToken.get_pad_token_value())

def _get_dataloader(songs: list[str], batch_size: int):
    dataset = MusicDataset(songs, song_partition_method=SongPartitionMethod.AllSubstrings)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def _train_model_with_parameters(
        model: TransformerModel, 
        train_loader: DataLoader,
        optimizer: optim.Adam,
        criterion: _Loss,
        parameters: TestingParameters
    ):
    train_losses, train_accuracies, best_model_file = train_transformer(
        model,
        train_loader,
        optimizer,
        criterion,
        parameters.epochs,
        print_status = True,
        save_mode = Config.args.save_mode,
        save_on_accuracy=True,
        model_name=parameters.model_name,
    )

    return train_losses, train_accuracies, best_model_file

def _evaluate_model_with_parameters(model: TransformerModel, model_output_directory: str, parameters: TestingParameters):
    
    output_music = generate_music(model, parameters.test_song)

    song_name = list(Songs.__dict__.keys())[list(Songs.__dict__.values()).index(parameters.test_song)]
    midi_output_file = f'{model_output_directory}/generate_{song_name}.midi'
    write_midi(output_music, midi_output_file)
    
    return output_music


def search_parameters(parameter_list: list[TestingParameters], experiment_name: str):
    model_results = []

    output_directory = f'parameter_search/out/{experiment_name}'

    for parameters in parameter_list:
        print(f'Experiment: {experiment_name} -> current model: {parameters.model_name}')

        if Config.args.verbose:
            print(f'[VERBOSE] With parameters: {parameters}')

        model_output_directory = f'{output_directory}/{parameters.model_name}'
        os.makedirs(model_output_directory, exist_ok=True)

        model = _get_model_with_parameters(parameters)
        train_loader = _get_dataloader(parameters.training_songs, batch_size=128)
        optimizer = _get_optimizer(model, parameters)
        criterion = _get_criterion()

        train_losses, train_accuracies, best_model_file = _train_model_with_parameters(
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

        test_song_output = _evaluate_model_with_parameters(model, model_output_directory, parameters)
        test_song_output_string = MusicToken.to_joined_string(test_song_output, join_string=' ')

        results = {
            'Loss': f'[{min(train_losses):.4f}, {max(train_losses):.4f}]  Final: {train_losses[-1]:.4f}',
            'Accuracy': f'[{format_percentage(min(train_accuracies))}, {format_percentage(max(train_accuracies))}]  Final: {format_percentage(train_accuracies[-1])}',
            'Test Song Output': test_song_output_string,
            'Model File': best_model_file,
            'Output Directory': model_output_directory,
        }

        with open(f'{model_output_directory}/results.json', 'w') as results_file:
            results_file.write(json.dumps(results, indent=2))

        model_results.append(results)

    return list(zip(parameter_list, model_results))

if __name__ == '__main__':
    # results = search_parameters(TEST_EXPERIMENT, 'TestExperiment')
    results = search_parameters(TWINKLE_TWINKLE_EXPERIMENT, 'TwinkleExperiment')

    print('-' * 10, 'Testing Results', '-' * 10)
    for parameters, result in results:
        print(f'{parameters.model_name}: {json.dumps(result, indent=1)}')
    print('-' * 37)
