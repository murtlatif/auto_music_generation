import torch
from torch import nn, optim, no_grad
from torch.utils.data.dataloader import DataLoader


def train(model: nn.Module, criterion: nn.CrossEntropyLoss, optimizer: optim.Optimizer, loader: DataLoader):
    model.train()

    epoch_loss = 0
    for batch_idx, batch in enumerate(loader):
        src, tgt = batch
        src, tgt = src.transpose(1, 0), tgt.transpose(1, 0)
        optimizer.zero_grad()
        output = model(src, tgt[:-1, :])
        n = output.shape[-1]
        loss = criterion(output.reshape(-1, n), tgt[1:, :].reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)
