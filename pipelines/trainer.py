from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from utils.statistics import perplexity, nll_loss


def train(data: Tuple[DataLoader, DataLoader, DataLoader], model: nn.Module, epochs: int, learning_rate: float,
          learning_rate_decay: float, max_grad: float) -> Tuple[nn.Module, Dict]:
    """
    model training loop
    :param data: tup of DataLoaders train, valid, test
    :param model: LSTM_Model
    :param epochs: number of epochs
    :param learning_rate: initial learning rate
    :param learning_rate_decay: learning rate decay factor
    :param max_grad: gradient clipped at this value
    :return: trained model and perplexity scores (per epoch for valid and final score for test)
    """
    train_loader, valid_loader, test_loader = data
    start_time = time.time()

    perplexity_scores = {
        'valid': [],
        'test': 0
    }

    total_words = 0
    print("Starting training.\n")
    batch_size = train_loader.batch_size

    for epoch in range(epochs):
        states = model.init_state(batch_size)

        if epoch + 1 > 5:
            learning_rate = learning_rate / learning_rate_decay

        for i, (x, y) in enumerate(train_loader):
            total_words += x.numel()
            model.zero_grad()
            states = model.detach_states(states)
            scores, states = model.forward(x, states)
            loss = nll_loss(scores=scores, targets=y)
            loss.backward()

            with torch.no_grad():
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                for param in model.parameters():
                    param -= learning_rate * param.grad

            if i % (len(train_loader) // 10) == 0:
                end_time = time.time()
                print("batch no = {:d} / {:d}, ".format(i, len(train_loader)) +
                      "train loss = {:.3f}, ".format(loss.item() / batch_size) +
                      "wps = {:d}, ".format(round(total_words / (end_time - start_time))) +
                      "dw.norm() = {:.3f}, ".format(norm) +
                      "lr = {:.3f}, ".format(learning_rate) +
                      "since beginning = {:d} mins, ".format(round((end_time - start_time) / 60)))

        model.eval()
        valid_perplexity = perplexity(data=valid_loader, model=model, batch_size=batch_size)
        perplexity_scores['valid'].append(valid_perplexity)
        print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch + 1, valid_perplexity))
        print("*************************************************\n")

    test_perp = perplexity(data=test_loader, model=model, batch_size=batch_size)
    perplexity_scores['test'] = test_perp
    print("Test set perplexity : {:.3f}".format(test_perp))

    print("Training is over.")

    return model, perplexity_scores