from data.dataset.music_dataset import MusicDataset
from model.transformer.transformer_baseline import TransformerModel
from torch import LongTensor, argmax, no_grad
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader


def validation(model: TransformerModel, criterion: _Loss, loader: DataLoader):
    """
    Performs validation on the model.

    Args:
        model (TransformerModel): The transformer model to run validation for
        criterion (_Loss): The loss function to evaluate based on
        loader (DataLoader): The data loader

    Returns:
        [type]: [description]
    """
    model.eval()
    epoch_loss = 0

    with no_grad():
        for batch_idx, (source, target) in enumerate(loader):
            # Omit the current target element when passing into the transformer
            output = model(source)

            num_features = output.shape[-1]

            # Flatten the input and output to compute loss
            flat_labels = target.reshape(-1)
            flat_output = output.reshape(-1, num_features)

            # Accumulate loss
            loss = criterion(flat_output, flat_labels)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate_note_sequence(model: TransformerModel, input_notes: str) -> list[list[str]]:
    """
    Evaluates and returns the output sequence given a set of input notes.

    Args:
        model (TransformerModel): The transformer model to use to evaluate
        input_notes (str): The input sequence to evaluate from

    Returns:
        list[list[str]]: The processed outputs
    """
    encoded_notes = MusicDataset.encode_notes(input_notes).unsqueeze(0)

    model.eval()
    with no_grad():
        output = model(encoded_notes)
        processed_output = TransformerModel.process_output(output)

    return processed_output


# def test(model, max_len=3, test_times=1):
#     model.eval()
#     with no_grad():

#         for test_num in range(test_times):

#             s = random.randint(1, 4998)
#             cpu_src = [(s + j) * 2 for j in range(max_len)]
#             src = LongTensor(cpu_src).unsqueeze(1)
#             tgt = [0] + [(s + j) * 2 + 1 for j in range(max_len)]
#             pred = [0]
#             for j in range(max_len):
#                 inp = LongTensor(pred).unsqueeze(1)
#                 output = model(src, inp)
#                 out_num = output.argmax(2)[-1].item()
#                 pred.append(out_num)
#             print("input: ", cpu_src)
#             print("target: ", tgt)
#             print("predict: ", pred)
