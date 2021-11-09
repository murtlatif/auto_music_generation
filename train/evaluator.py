from data.notes.simple_notes_dataset import SimpleNotes
from model.transformer.transformer_baseline import TransformerModel
from torch import LongTensor, no_grad, argmax


def validation(model, criterion, loader):
    model.eval()
    epoch_loss = 0
    with no_grad():
        for i, batch in enumerate(loader):
            src, tgt = batch
            src, tgt = src.transpose(1, 0), tgt.transpose(1, 0)
            output = model(src, tgt[:-1, :])
            n = output.shape[-1]
            loss = criterion(output.reshape(-1, n), tgt[1:, :].reshape(-1))
            epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate_note_sequence(model: TransformerModel, input_notes: str):
    """
    Evaluates and returns the output sequence given a set of input notes.

    Args:
        model (TransformerModel): The transformer model to use to evaluate
        input_notes (str): The input sequence to evaluate from

    Returns:
        [type]: [description]
    """
    encoded_notes = SimpleNotes.encode_notes(input_notes)

    model.eval()
    with no_grad():

        source = LongTensor([encoded_notes]).transpose(1, 0)
        target = LongTensor([[0] + encoded_notes]).transpose(1, 0)

        output = model(source, target[:-1, :])

        # Get the maximally ranked note
        # output_indices = argmax(output, axis=2)

    return output


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
