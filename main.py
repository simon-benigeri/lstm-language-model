from lstm import LSTM_Model
from data import init_datasets
import numpy as np
import torch
import torch.nn as nn
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


#The loss function.
def _neg_log_likelihood_loss(scores, y):
    batch_size = y.size(1)
    print("y shape: ", y.shape)
    print("scores shape: ", scores.shape)
    expscores = scores.exp()
    print("expscores shape: ", expscores.shape)
    probabilities = expscores / expscores.sum(1, keepdim = True)
    print("prob shape: ", probabilities.shape)
    answerprobs = probabilities[range(len(y.reshape(-1))), y.reshape(-1)]
    #I multiply by batch_size as in the original paper
    #Zaremba et al. sum the loss over batches but average these over time.
    return torch.mean(-torch.log(answerprobs) * batch_size)

def neg_log_likelihood_loss(scores, targets):
    # substituting with cross entropy loss
    batch_size = targets.size(1)
    return F.cross_entropy(scores.reshape(-1, scores.size(2)), targets.reshape(-1)) * batch_size


def get_perplexity(model, data, batch_size):
    with torch.no_grad():
        losses = []
        states = model.init_state(batch_size)
        for x, y in data:
            x = torch.transpose(x, 0, 1)
            y = torch.transpose(y, 0, 1)
            scores, states = model(x, states)
            loss = neg_log_likelihood_loss(scores, y)
            #Again with the sum/average implementation described in 'nll_loss'.
            losses.append(loss.data.item() / batch_size)
    return np.exp(np.mean(losses))


def train(model, data, epochs, learning_rate, learning_rate_decay, max_grad):
    
    train_loader, valid_loader, test_loader = data
    start_time = time.time()
    
    total_words = 0
    print("Starting training.\n")
    
    for epoch in range(epochs):
        batch_size = train_loader.batch_size
        states = model.init_state(batch_size)

        if epoch > 5:
            learning_rate = learning_rate / learning_rate_decay
        
        for i, (x, y) in enumerate(train_loader):
            # print(f"x size before : {x.size()}")
            # x = torch.transpose(x, 0, 1)
            # print(f"x size after : {x.size()}")
            # y = torch.transpose(y, 0, 1)
            total_words += x.numel()
            model.zero_grad()
            
            # batch_size = len(x))
            states = model.detach(states)
            scores, states = model(x, states)
            loss = neg_log_likelihood_loss(scores, y)
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
        valid_perplexity = get_perplexity(model, valid_loader, batch_size)
        print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch+1, valid_perplexity))
        print("*************************************************\n")
    test_perp = get_perplexity(model, test_loader, batch_size)
    print("Test set perplexity : {:.3f}".format(test_perp))
    print("Training is over.")
    return model


def main():
    hyperparams = {
        'embed_dims': None,
        'freq_threshold': 3,
        'dropout_prob': 0.5,
        'init_range': 0.05,
        'epochs': 40,
        'learning_rate': 1,
        'learning_rate_decay': 1.2,
        'num_layers': 2,
        'batch_size': 20,
        'time_steps': 35,
        'max_grad': 5,
        'embed_tying': False,
        'bias': False,
        'save_model': True,
        'load_model': True,
        'model_path': 'lstm_model',
        'topic': 'wiki', # enter 'wiki' or 'nyt_covid'
        'path': 'data/corpora'
    }

    data_params = {k:hyperparams for k in ['topic', 'path', 'freq_threshold', 'time_steps', 'batch_size']}
    datasets = init_datasets(**data_params)

    model_params = ['embed_dims', 'dropout_prob', 'init_range', 'num_layers', 'step_size', 'max_grad', 'embed_tying', 'bias']
    model_params = {k:hyperparams[k] for k in model_params}
    model = LSTM_Model(**model_params)

    if hyperparams['load_model']:
        model.load_state_dict(torch.load(hyperparams['model_path']))

    else:
        training_params = ['epochs', 'learning_rate', 'learning_rate_decay', 'max_grad']
        training_params = {k:hyperparams[k] for k in training_params}
        model = train(model, datasets['data_loaders'], **training_params)

    # now calculate perplexities for train, valid, test
    for d, l in zip(datasets, ['train', 'valid', 'test']):
        perplexity = get_perplexity(d, model, d.batch_size)
        print("perplexity on %s dataset: %.3f, " % (l, perplexity))

    if hyperparams['save_model']:
        torch.save(model.state_dict(), hyperparams['model_path'])