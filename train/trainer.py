import torch
from model.transformer.transformer_baseline import TransformerModel
from torch import nn, no_grad, optim
from torch.utils.data.dataloader import DataLoader


def train_one_epoch_transformer(transformer: TransformerModel, criterion: nn.CrossEntropyLoss, optimizer: optim.Optimizer, loader: DataLoader, status_print_interval: int = 100) -> float:

    transformer.train()

    epoch_loss = 0

    for batch_idx, (source, target) in enumerate(loader):
        source, target = source.transpose(1, 0), target.transpose(1, 0)

        optimizer.zero_grad()

        # Omit the current target element when passing into the transformer
        observed_target = target[:-1, :]

        output = transformer(source, observed_target)
        num_features = output.shape[-1]

        # Omit the first element when using the target as a label
        target_as_label = target[1:, :]

        # Flatten the input and output to compute loss
        flat_labels = target_as_label.reshape(-1)
        flat_output = output.reshape(-1, num_features)

        loss = criterion(flat_output, flat_labels)
        loss.backward()

        print(f'Output: {flat_output}, Labels: {flat_labels}, Loss: {loss}')

        # Clip the gradient by the norm to prevent exploding gradient
        nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)
