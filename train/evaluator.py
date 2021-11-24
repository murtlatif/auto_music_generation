from data.dataset.music_token import MusicToken
from model.transformer.transformer_baseline import TransformerModel
from torch import nn, no_grad
from torch.utils.data.dataloader import DataLoader


def evaluate_note_sequence(model: TransformerModel, input_notes: str) -> list[list[MusicToken]]:
    """
    Evaluates and returns the output sequence given a set of input notes.

    Args:
        model (TransformerModel): The transformer model to use to evaluate
        input_notes (str): The input sequence to evaluate from

    Returns:
        list[list[str]]: The processed outputs
    """
    encoded_notes = MusicToken.to_tensor(MusicToken.from_string(input_notes)).unsqueeze(0)

    model.eval()
    with no_grad():
        output = model(encoded_notes)
        processed_output = TransformerModel.process_output(output)

    return processed_output


def validate_transformer_single_epoch(model: TransformerModel, criterion: nn.CrossEntropyLoss, loader: DataLoader):
    """
    CURRENTLY UNUSED. No longer using validation in training process.

    Performs validation on the TransformerModel for a single epoch.

    Args:
        model (TransformerModel): The transformer model to run validation for
        criterion (nn.CrossEntropyLoss): The loss function to evaluate based on
        loader (DataLoader): The data loader

    Returns:
        tuple[float, float]: The validation loss and accuracy
    """

    model.eval()

    epoch_total_loss = 0
    epoch_correct = 0
    epoch_predictions = 0

    with no_grad():
        for batch_idx, (source, target) in enumerate(loader):

            output = model(source)

            num_features = output.shape[-1]

            # Flatten the input and output to compute loss
            flat_labels = target.reshape(-1) - 1
            flat_output = output.reshape(-1, num_features)

            # Accumulate loss
            loss = criterion(flat_output, flat_labels)

            # Compute epoch metrics
            epoch_total_loss += loss.item()

            output_notes = output.argmax(axis=-1)
            correct_notes = output_notes == target
            epoch_correct += correct_notes.sum()
            epoch_predictions += len(correct_notes)

        epoch_loss = epoch_total_loss / len(loader)
        epoch_accuracy = epoch_correct / epoch_predictions

    return epoch_loss, epoch_accuracy
