from timeit import default_timer as timer

import torch
from automusicgen.config import Config
from automusicgen.data.dataset.music_token import MusicToken
from automusicgen.io.model_file_manager import (save_model,
                                                save_model_parameters,
                                                save_model_state)
from automusicgen.model.transformer.transformer_baseline import \
    TransformerModel
from automusicgen.parameter_search.testing_parameters import \
    TransformerParameters
from automusicgen.util.constants import SaveMode
from automusicgen.util.device import get_device
from automusicgen.util.string_formatter import format_percentage
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader


def train_transformer(
    model: TransformerModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: _Loss,
    parameters: TransformerParameters = None,
):
    """
    Trains a transformer model for multiple epochs.

    Args:
        model (TransformerModel): Model to train
        train_loader (DataLoader): DataLoader for the training set
        optimizer (optim.Optimizer): Optimizer for the model
        criterion (_Loss): Loss function
        epochs (int, optional): Number of epochs to train. Defaults to 10.
        print_status (bool, optional): If it should print a status update every epoch. Defaults to True.
        save_best_model (bool, optional): If it should save the model every time it outperforms the best training loss. Defaults to True.
        save_on_accuracy (bool, optinoal): If the outperforming metric should instead be accuracy. Defaults to False.

    Returns:
        tuple[list[float], list[float], str]: Train loss per epoch, and the best model file name if one was saved.
    """
    model_name = parameters.model_name
    epochs = parameters.epochs
    print_status = parameters.print_status
    save_mode = parameters.save_mode
    save_on_accuracy = parameters.save_on_accuracy
    
    train_loss_per_epoch = []
    train_accuracy_per_epoch = []
    best_epoch = -1
    best_train_loss = None
    best_train_accuracy = None
    best_model_file_name = ''
    best_model_param_file_name = ''
    best_model_state = None

    def save_current_model(by_state: bool = False, save_parameters: bool = True):
        if (not best_train_accuracy) and (not best_train_loss):
            model_tag = f'untrained'
        else:
            model_tag = f'acc_{best_train_accuracy:.2f}' if save_on_accuracy else f'loss_{best_train_loss:.3f}'

        model_device = get_device()
        model_save_file = f'tfmr_{model_name}_{model_device}_{model_tag}.pt'

        if by_state:
            save_model_state(model_save_file, best_model_state)
        else:
            save_model(model_save_file, model)

        parameters_save_file = f'tfmr_{model_name}_{model_device}_{model_tag}.params'
        if save_parameters:
            save_model_parameters(parameters_save_file, parameters)

        return model_save_file, parameters_save_file

    try:
        for epoch in range(epochs):

            # Record time taken for the epoch
            start_time = timer()

            epoch_train_loss, epoch_train_accuracy = train_transformer_single_epoch(
                model, criterion, optimizer, train_loader)

            end_time = timer()

            # Add the train loss/accuracy for a plot
            train_loss_per_epoch.append(epoch_train_loss)
            train_accuracy_per_epoch.append(epoch_train_accuracy)

            # Record the best models
            is_new_best = False

            if save_on_accuracy and (best_train_accuracy is None or epoch_train_accuracy > best_train_accuracy):
                is_new_best = True
                best_train_accuracy = epoch_train_accuracy
                best_epoch = epoch

            if (not save_on_accuracy) and (best_train_loss is None or epoch_train_loss < best_train_loss):
                is_new_best = True
                best_train_loss = epoch_train_loss
                best_epoch = epoch

            if print_status:
                status_text = f'Epoch {epoch+1:3}/{epochs} ({format_percentage((epoch+1)/epochs)}) | ' \
                              f'Loss: {epoch_train_loss:.4f} | Accuracy: {format_percentage(epoch_train_accuracy)} | ' \
                              f'Time Taken: {end_time - start_time:.3f}s'

                if is_new_best:
                    status_text += ' [NEW BEST!]'

                print(status_text)

            if is_new_best:
                if save_mode == SaveMode.SaveEachNewBest:
                    best_model_file_name, best_model_param_file_name = save_current_model()

                elif save_mode == SaveMode.SaveBest:
                    best_model_state = model.state_dict()

        completed_text = 'Completed training.'
        if save_on_accuracy:
            completed_text += f' Best training accuracy was {format_percentage(best_train_accuracy)} on epoch {best_epoch}.'
        else:
            completed_text += f' Best training loss was {best_train_loss:.4f} on epoch {best_epoch}.'

        print(completed_text)

    except KeyboardInterrupt:
        abort_text = f'Aborted training. Done {epoch}/{epochs} epochs.'
        print(abort_text)
    except Exception:
        raise
    finally:
        if save_mode == SaveMode.SaveBest:
            best_model_file_name, best_model_param_file_name = save_current_model(by_state=True)
            print(f'Best model saved on epoch {best_epoch} as: {best_model_file_name}')

        elif save_mode == SaveMode.SaveLast:
            best_model_file_name, best_model_param_file_name = save_current_model()
            print(f'Last model saved as: {best_model_file_name}')

    return train_loss_per_epoch, train_accuracy_per_epoch, best_model_file_name, best_model_param_file_name


def train_transformer_single_epoch(
    transformer: TransformerModel,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    print_progress_interval: int = 100,
) -> float:
    """
    Trains a TransformerModel with one epoch.

    Args:
        transformer (TransformerModel): The transformer model to train
        criterion (nn.CrossEntropyLoss): The criterion to evaluate the loss with
        optimizer (optim.Optimizer): The optimizer used to train the parameters
        loader (DataLoader): The dataset loader to train on

    Returns:
        tuple[float, float]: The loss and accuracy of the epoch
    """

    transformer.train()

    epoch_total_loss = 0
    epoch_predictions = 0
    epoch_correct = 0

    device = get_device()

    total_num_batches = len(loader)

    for batch_idx, (source, target) in enumerate(loader):
        optimizer.zero_grad()

        # Expect source to be of shape (N, S) and target to be of shape (N, T)
        source = source.to(device=device)
        target = target.to(device=device)

        # Cut the source and target to be at most the amount the positional encoder can take
        max_bandwidth = transformer.positional_encoder.max_bandwidth()

        if source.size(1) > max_bandwidth:
            source = source[:, -max_bandwidth:]

        target_input = target[:, :-1]
        if target_input.size(1) > max_bandwidth:
            target_input = target[:, -max_bandwidth:]

        source_mask, target_mask, source_padding_mask, target_padding_mask = transformer.create_mask(source, target_input)

        output = transformer(source, target_input, source_mask, target_mask, source_padding_mask, target_padding_mask)
        num_features = output.shape[-1]

        # Flatten the input and output to compute loss
        output = output.reshape(-1, num_features)
        target_output = target[:, 1:]
        flattened_target_output = target_output.flatten()

        if Config.args.verbose > 1:
            verbosity = Config.args.verbose
            print(f'Sample outputs: {output[:verbosity - 1]} with target: {flattened_target_output[:verbosity - 1]}')

        loss = criterion(output, flattened_target_output)
        loss.backward()

        optimizer.step()

        # Compute epoch metrics
        epoch_total_loss += loss.item()

        output_notes = output.argmax(axis=-1)
        correct_notes = output_notes == flattened_target_output

        # Do not count padded indices
        correct_notes_excluding_pads = torch.where(output_notes != transformer.PAD_TOKEN, correct_notes, False)

        num_correct_notes = correct_notes_excluding_pads.sum().item()
        num_predictions = len(flattened_target_output[flattened_target_output != transformer.PAD_TOKEN])
        epoch_correct += num_correct_notes
        epoch_predictions += num_predictions

        if Config.args.verbose > 0 and batch_idx % print_progress_interval == 0:
            status_text = f'\tBatch #{batch_idx+1:3}/{total_num_batches} ({format_percentage((batch_idx+1)/total_num_batches)}) | ' \
                          f'Batch Loss: {loss.item():.4f} | Batch Correct: {num_correct_notes}/{num_predictions} ({format_percentage((num_correct_notes/num_predictions))})| ' \
                          f'Loss So Far: {epoch_total_loss:.4f} | Correct So Far: {epoch_correct}/{epoch_predictions} ({format_percentage((num_correct_notes/num_predictions))}) | '

            print(status_text)
    epoch_loss = epoch_total_loss / len(loader)
    epoch_accuracy = epoch_correct / epoch_predictions

    return epoch_loss, epoch_accuracy
