from src.lstm import LSTM_Model
from src.datasets import init_datasets
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

# (i) embedding space dimensionality,
# (ii) vocabulary frequency threshold,
# (iii) a keep probability for dropout,
# (iv) an initialization range,
# (v) number of training epochs,
# (vi) an initial learning rate and decay schedule,
# (vii) number of layers,
# (viii) batch size,
# (ix) step size,
# (x) maximum gradient,
# (xi) embedding tying,
# (xii) a bias flag, and any others that you think may be necessary.


def neg_log_likelihood_loss(scores, targets):
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


def get_perplexity(data:DataLoader, model:nn.Module, batch_size:int) -> float:
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
            loss = neg_log_likelihood_loss(scores, y)
            losses.append(loss.data.item() / batch_size)

    perplexity = np.exp(np.mean(losses))

    return perplexity


def train(data:Tuple[DataLoader, DataLoader, DataLoader], model:nn.Module, epochs:int, learning_rate:float,
          learning_rate_decay:float, max_grad:float) -> Tuple[nn.Module, Dict]:
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
            # scores, states = model(x, states)
            loss = neg_log_likelihood_loss(scores=scores, targets=y)
            loss.backward()
            
            with torch.no_grad():
                norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad)
                for param in model.parameters():
                    param -= learning_rate * param.grad
            
            if i % (len(train_loader) // 10) == 0:
                end_time = time.time()
                print("batch no = {:d} / {:d}, ".format(i, len(train_loader)) +
                      "train loss = {:.3f}, ".format(loss.item() / batch_size) +
                      "wps = {:d}, ".format(round(total_words/(end_time-start_time))) +
                      "dw.norm() = {:.3f}, ".format(norm) +
                      "lr = {:.3f}, ".format(learning_rate) +
                      "since beginning = {:d} mins, ".format(round((end_time-start_time)/60))) # +
                      # "cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated()/1024/1024/1024))
        
        model.eval()
        valid_perplexity = get_perplexity(data=valid_loader, model=model, batch_size=batch_size)
        perplexity_scores['valid'].append(valid_perplexity)
        print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch+1, valid_perplexity))
        print("*************************************************\n")

    test_perp = get_perplexity(data=test_loader, model=model, batch_size=batch_size)
    perplexity_scores['test'] = test_perp
    print("Test set perplexity : {:.3f}".format(test_perp))

    print("Training is over.")

    return model, perplexity_scores


def main():
    hyperparams = {
        'embed_dims': 200,
        'device': 'cpu', # 'gpu'
        'freq_threshold': 3,
        'dropout_prob': 0.5,
        'init_range': 0.05,
        'epochs': 20,
        'learning_rate': 1,
        'learning_rate_decay': 1.25,
        'num_layers': 2,
        'batch_size': 20,
        'time_steps': 30,
        'max_grad': 2.0,
        'embed_tying': False,
        'bias': True,
        'save_model': True,
        'load_model': False,
        'model_path': 'lstm_model',
        'topic': 'wiki', # enter 'wiki' or 'nyt_covid'
        'path': '../data/corpora'
    }

    # set params for init_datasets
    data_params = {k:hyperparams[k] for k in ['topic','freq_threshold', 'time_steps', 'batch_size',  'path']}
    datasets = init_datasets(**data_params)

    # get vocab size and word2index dict
    vocab_size = datasets['vocab_size']
    word2index = datasets['word2index']

    # get the data_loaders: train, valid, test
    data_loaders = datasets['data_loaders']

    # set params for model training
    model_params = ['device', 'embed_dims', 'dropout_prob', 'init_range',
                    'num_layers', 'max_grad', 'embed_tying', 'bias']
    model_params = {k:hyperparams[k] for k in model_params}
    model_params['vocab_size'] = vocab_size

    # create model
    model = LSTM_Model(**model_params)

    # as a sanity check, we print out vocab size and perplexity on validation set
    print(f"vocab size : {vocab_size}")
    # data_loaders := (train, valid, test)
    perplexity = get_perplexity(data=data_loaders[1], model=model, batch_size=data_loaders[1].batch_size)
    print("perplexity on %s dataset before training: %.3f, " % ('valid', perplexity))

    # load model
    if hyperparams['load_model']:
        model.load_state_dict(torch.load(hyperparams['model_path']))
    else:
        # set training parameters
        training_params = ['epochs', 'learning_rate', 'learning_rate_decay', 'max_grad']
        training_params = {k:hyperparams[k] for k in training_params}
        # launch model training
        model, perplexity_scores = train(data=data_loaders, model=model, **training_params)

    # now calculate perplexities for train, valid, test
    for d, l in zip(data_loaders, ['train', 'valid', 'test']):
        perplexity = get_perplexity(data=d, model=model, batch_size=d.batch_size)
        print("perplexity on %s dataset after training : %.3f, " % (l, perplexity))

    # save model
    if hyperparams['save_model']:
        torch.save(model.state_dict(), hyperparams['model_path'])

if __name__=='__main__':
    main()