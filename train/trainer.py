from config import Config
from model.transformer.transformer_baseline import TransformerModel
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader
from util.model_file_manager import save_model


def train_transformer(
    model: TransformerModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: _Loss,
    epochs: int = 10,
    print_status: bool = True,
    save_best_model: bool = True,
    save_on_accuracy: bool = False,
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
    train_loss_per_epoch = []
    train_accuracy_per_epoch = []
    best_epoch = -1
    best_train_loss = None
    best_train_accuracy = None
    best_model_name = ''

    try:
        for epoch in range(epochs):
            epoch_train_loss, epoch_train_accuracy = train_transformer_single_epoch(
                model, criterion, optimizer, train_loader)

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
                status_text = f'Epoch {epoch+1:3}/{epochs} ({100*(epoch+1)/epochs:5.1f}%) | ' \
                              f'Loss (Train): {epoch_train_loss:.4f} | Accuracy (Train): {epoch_train_accuracy:5.2%}'

                if is_new_best:
                    status_text += ' [NEW BEST!]'

                print(status_text)

            if save_best_model and is_new_best:
                model_tag = f'acc_{best_train_accuracy:.2f}' if save_on_accuracy else f'loss_{best_train_loss:.3f}'
                model_name = Config.args.name or 'Unnamed'
                model_save_file = f'tfmr_{model_name}_{model_tag}.pt'

                save_model(model_save_file, model)
                best_model_name = model_save_file

        completed_text = 'Completed training.'
        if save_on_accuracy:
            completed_text += f' Best training accuracy was {best_train_accuracy} on epoch {best_epoch}.'
        else:
            completed_text += f' Best training loss was {best_train_loss} on epoch {best_epoch}.'

        if save_best_model:
            completed_text += f' Best model saved on epoch {best_epoch} as: {best_model_name}'
        print(completed_text)

    except KeyboardInterrupt:
        abort_text = f'Aborted training. Done {epoch}/{epochs} epochs.'
        if save_best_model:
            abort_text += f' Best model saved on epoch {best_epoch} as: {best_model_name}'

        print(abort_text)

    return train_loss_per_epoch, train_accuracy_per_epoch, best_model_name


def train_transformer_single_epoch(
    transformer: TransformerModel,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    loader: DataLoader,
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

    for batch_idx, (source, target) in enumerate(loader):
        optimizer.zero_grad()

        output = transformer(source)
        num_features = output.shape[-1]

        # Flatten the input and output to compute loss
        output = output.reshape(-1, num_features)
        target = target.flatten()

        if Config.args.verbose:
            verbosity = Config.args.verbose
            print(f'Sample outputs: ', output[:verbosity], target[:verbosity])

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()

        # Compute epoch metrics
        epoch_total_loss += loss.item()

        output_notes = output.argmax(axis=-1)
        correct_notes = output_notes == target
        epoch_correct += correct_notes.sum()
        epoch_predictions += len(correct_notes)

    epoch_loss = epoch_total_loss / len(loader)
    epoch_accuracy = epoch_correct / epoch_predictions

    return epoch_loss, epoch_accuracy
