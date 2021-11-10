from config import Config
from model.transformer.transformer_baseline import TransformerModel
from torch import nn, optim
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader
from util.model_file_manager import save_model

from .evaluator import validation


def train_transformer(
    model: TransformerModel,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: _Loss,
    epochs: int = 10,
    print_status: bool = True,
    save_best_model: bool = True,
):
    """
    Trains a transformer model for multiple epochs.

    Args:
        model (TransformerModel): Model to train
        train_loader (DataLoader): DataLoader for the training set
        validation_loader (DataLoader): DataLoader for the validation set
        optimizer (optim.Optimizer): Optimizer for the model
        criterion (_Loss): Loss function
        epochs (int, optional): Number of epochs to train. Defaults to 10.
        print_status (bool, optional): If it should print a status update every epoch. Defaults to True.
        save_best_model (bool, optional): If it should save the model every time it outperforms the best validation loss. Defaults to True.

    Returns:
        tuple[list, list, str]: Train & validation loss per epoch, and the best model file name if one was saved.
    """
    train_loss_per_epoch = []
    validation_loss_per_epoch = []

    best_validation_loss = float('inf')
    best_epoch = -1
    best_model_name = ''

    try:
        for epoch in range(epochs):
            epoch_train_loss = train_one_epoch_transformer(model, criterion, optimizer, train_loader)
            epoch_validation_loss = validation(model, criterion, validation_loader)

            # Add the train/validation losses for a plot
            train_loss_per_epoch.append(epoch_train_loss)
            validation_loss_per_epoch.append(epoch_validation_loss)

            # scheduler.step()

            # Record the best models
            is_new_best = False
            if epoch_validation_loss < best_validation_loss:
                is_new_best = True
                best_validation_loss = epoch_validation_loss
                best_epoch = epoch

            if print_status:
                status_text = f'Epoch {epoch+1:3}/{epochs} ({100*(epoch+1)/epochs:5.1f}%) | Loss (Train): {epoch_train_loss:.4f}, Loss (Validation): {epoch_validation_loss:.4f}'

                if is_new_best:
                    status_text += ' [NEW BEST!]'

                print(status_text)

            if save_best_model and is_new_best:
                model_name = f'transformer_{Config.args.name or "Unnamed"}_{epoch_validation_loss:.4f}.pt'
                save_model(model_name, model)
                best_model_name = model_name

        completed_text = f'Completed training. Best validation loss was {best_validation_loss} on epoch {best_epoch}.'
        if save_best_model:
            completed_text += f' Best model saved on epoch {best_epoch} as: {best_model_name}'
        print(completed_text)

    except KeyboardInterrupt:
        abort_text = f'Aborted training. Done {epoch}/{epochs} epochs.'
        if save_best_model:
            abort_text += f' Best model saved on epoch {best_epoch} as: {best_model_name}'

        print(abort_text)

    return train_loss_per_epoch, validation_loss_per_epoch, best_model_name


def train_one_epoch_transformer(transformer: TransformerModel, criterion: nn.CrossEntropyLoss, optimizer: optim.Optimizer, loader: DataLoader) -> float:
    """
    Trains a transformer model for one epoch.

    Args:
        transformer (TransformerModel): The transformer model to train
        criterion (nn.CrossEntropyLoss): The criterion to evaluate the loss with
        optimizer (optim.Optimizer): The optimizer used to train the parameters
        loader (DataLoader): The dataset loader to train on

    Returns:
        float: The loss of the epoch given by the criterion 
    """

    transformer.train()

    epoch_loss = 0

    for batch_idx, (source, target) in enumerate(loader):
        optimizer.zero_grad()

        output = transformer(source)

        output = output.reshape(output.shape[0] * output.shape[1], -1)
        target = target.flatten()

        loss = criterion(output, target)
        loss.backward()

        # Clip the gradient by the norm to prevent exploding gradient
        nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)
