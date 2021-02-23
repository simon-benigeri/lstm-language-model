from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

def nll_loss(scores, targets):
    """
    :param scores: shape (batch_size, time_steps, vocab_size)
    :param targets: shape (batch_size, time_steps)
    :return: cross entropy loss, takes input (N, C) and (N) where C=num classes (words in vocab)
    """
    # targets are shape (batch_size, time_steps)
    batch_size = targets.size(0)
    # scores: (batch_size, time_steps, vocab_size)(batch_size * time_steps, vocab_size)
    scores = scores.reshape(-1, scores.size(2))
    # targets: (batch_size, time_steps) -> (batch_size*time_steps)
    targets = targets.reshape(-1)

    return F.cross_entropy(input=scores, target=targets) * batch_size


def perplexity(data:DataLoader, model:nn.Module, batch_size:int) -> float:
    """
    :param data:
    :param model:
    :param batch_size:
    :return:
    """
    model.eval()
    with torch.no_grad():
        losses = []
        states = model.init_state(batch_size)
        for (x, y) in data:
            # scores, states = model(x, states)
            scores, states = model.forward(x=x, states=states)
            loss = nll_loss(scores, y)
            losses.append(loss.data.item() / batch_size)

    perplexity = np.exp(np.mean(losses))

    return perplexity