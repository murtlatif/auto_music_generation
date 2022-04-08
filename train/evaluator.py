import random

import torch
from automusicgen.config import Config
from automusicgen.data.dataset.music_token import MusicToken
from automusicgen.data.tokenize import midi_tokenizer
from automusicgen.model.transformer.transformer_baseline import \
    TransformerModel
from automusicgen.util.device import get_device
from torch import LongTensor, Tensor, nn, no_grad
from torch.utils.data.dataloader import DataLoader


def _greedy_decode(model: TransformerModel, source: Tensor, source_mask: Tensor, max_prediction_length: int, start_token: int, end_token: int):
    DEVICE = get_device()

    max_target_size = Config.args.max_len
    source = source.to(DEVICE)
    source_mask = source_mask.to(DEVICE)
    current_source = source
    current_source_mask = source_mask

    source_left_idx_upper_limit = max(source.size(0) - max_target_size, 0)
    source_right_idx_lower_limit = min(max_target_size, source.size(0))
    # if source.size(0) > max_target_size:
    #     truncated_source = source[-max_target_size:]
    #     truncated_source_mask = source_mask[-max_target_size:, -max_target_size:]
    if source.size(0) > max_target_size:
            source_left_idx = 0
            source_right_idx = max(source_left_idx + max_target_size, source_right_idx_lower_limit)
            current_source = source[source_left_idx:source_right_idx]
            current_source_mask = source_mask[source_left_idx:source_right_idx, source_left_idx:source_right_idx]
            memory = model.encode(source=current_source, source_mask=current_source_mask)
    else:
        memory = model.encode(source=source, source_mask=source_mask)

    output = torch.ones(1, 1).fill_(start_token).type(torch.int).to(DEVICE)
    current_target = torch.ones(1, 1).fill_(start_token).type(torch.int).to(DEVICE)

    for prediction_idx in range(max_prediction_length - 1):
        if current_target.size(0) > max_target_size:
            current_target = torch.ones(1, 1).type_as(source.data).fill_(current_target[-1].item())
            # current_target = output[-max_target_size:]
            # random_idx = random.randint(1, max_target_size)
            # current_target = output[-random_idx:]
        # TODO: Add source window
        if source.size(0) > max_target_size:
            source_left_idx = min(prediction_idx, source_left_idx_upper_limit)
            source_right_idx = max(source_left_idx + max_target_size, source_right_idx_lower_limit)
            current_source = source[source_left_idx:source_right_idx]
            current_source_mask = source_mask[source_left_idx:source_right_idx, source_left_idx:source_right_idx]
            memory = model.encode(source=current_source, source_mask=current_source_mask)

        memory = memory.to(DEVICE)
        target_mask = model.transformer.generate_square_subsequent_mask(current_target.size(0)).to(DEVICE)
        decoded_output = model.decode(target=current_target, memory=memory, target_mask=target_mask)
        # target_mask = model.transformer.generate_square_subsequent_mask(output.size(0)).to(DEVICE)
        # decoded_output = model.decode(target=output, memory=memory, target_mask=target_mask)
        decoded_output = decoded_output.transpose(0, 1)
        next_note_probabilities = model.fc_out(decoded_output[:, -1])
        _, next_note = torch.max(next_note_probabilities, dim=1)
        next_note = next_note.item()

        output = torch.cat([
            output,
            torch.ones(1, 1).type_as(source.data).fill_(next_note)
        ], dim=0)

        current_target = torch.cat([
            current_target,
            torch.ones(1, 1).type_as(source.data).fill_(next_note)
        ])

        if next_note == end_token:
            break

    return output

def generate_music(model: TransformerModel, input_song: str) -> list[MusicToken]:
    model.eval()

    # Add <bos> and <eos> tags to the song
    tagged_input_song = f'<{input_song}>'

    # Format the music tokens as (S, N) where S is sequence length, N is batch size
    source = MusicToken.to_tensor(MusicToken.from_string(tagged_input_song)).view(-1, 1)[:model.positional_encoder.max_bandwidth()]
    num_tokens = source.shape[0]
    source_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)

    target_tokens = _greedy_decode(
        model,
        source,
        source_mask,
        max_prediction_length=num_tokens+10,
        start_token=MusicToken.BeginningOfSequence.value,
        end_token=MusicToken.EndOfSequence.value,
    ).flatten().tolist()

    output_music_tokens = [MusicToken(target_token) for target_token in target_tokens]
    return output_music_tokens

def generate_music_midi(model: TransformerModel, input_song: list[int], sos_token: int, eos_token: int):
    model.eval()

    max_song_length = model.positional_encoder.max_bandwidth()
    # truncated_input_song = input_song[:max_song_length]
    # input_song_tensor = LongTensor(truncated_input_song).to(device=get_device())
    input_song_tensor = LongTensor(input_song).to(device=get_device())

    # Format the music tokens as (S, N) where S is sequence length, N is batch size
    source = input_song_tensor.view(-1, 1)

    num_tokens = source.shape[0]
    source_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)

    target_tokens = _greedy_decode(
        model,
        source,
        source_mask,
        max_prediction_length=num_tokens+500,
        start_token=sos_token,
        end_token=eos_token,
    ).flatten().tolist()

    output_music_tokens = target_tokens

    return output_music_tokens


def evaluate_note_sequence(model: TransformerModel, input_notes: str):
    """
    DEPRECATED

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
        processed_output = model.process_output(output)

    return processed_output

def evaluate_note_sequence_iteratively(model: TransformerModel, input_notes: str, number_of_iterations: int):
    """
    DEPRECATED

    Evaluates the output sequence, then evaluates the output sequence again
    using the previous output as the new input. Repeats for the given number
    of iterations.
    """
    for i in range(number_of_iterations):
        output_sequence = evaluate_note_sequence(model, input_notes)
        output_notes_string = MusicToken.to_joined_string(output_sequence)
        input_notes = input_notes + output_notes_string

    return input_notes


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
