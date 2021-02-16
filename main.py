from lstm import LSTM_Model
from data import init_datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    # print("y shape: ", y.shape)
    # print("scores shape: ", scores.shape)
    expscores = scores.exp()
    # print("expscores shape: ", expscores.shape)
    probabilities = expscores / expscores.sum(1, keepdim = True)
    # print("prob shape: ", probabilities.shape)
    answerprobs = probabilities[range(len(y.reshape(-1))), y.reshape(-1)]
    #I multiply by batch_size as in the original paper
    #Zaremba et al. sum the loss over batches but average these over time.
    return torch.mean(-torch.log(answerprobs) * batch_size)

def neg_log_likelihood_loss(scores, targets):
    # substituting with cross entropy loss
    # get batch size
    batch_size = targets.size(1)

    # print(f"scores size : {scores.size()}")
    # print(f"scores reshaped to : {scores.reshape(-1, scores.size(2)).size()}")

    # print(f"targets size : {targets.size()}")
    # print(f"targets reshaped to : {targets.reshape(-1).size()}")

    # scores are shape (batch_size, time_steps, vocab_size)
    # scores are reshaped to (batch_size * time_steps, vocab_size)

    # targets are shape (batch_size, time_steps)
    # targets are reshapes to (batch_size*time_steps)
    return F.cross_entropy(scores.reshape(-1, scores.size(2)), targets.reshape(-1)) * batch_size


def get_perplexity(data, model, batch_size):
    model.eval()
    with torch.no_grad():
        losses = []
        states = model.init_state(batch_size)
        for x, y in data:
            # print(f"x size : {x.size()}")
            # print(f"y size : {y.size()}")
            scores, states = model(x, states)

            # print(f"scores size : {scores.size()}")
            loss = neg_log_likelihood_loss(scores, y)

            # print(f"loss : {loss}")
            #Again with the sum/average implementation described in 'nll_loss'.
            losses.append(loss.data.item() / batch_size)
    return np.exp(np.mean(losses))


def train(data, model, epochs, learning_rate, learning_rate_decay, max_grad):
    
    train_loader, valid_loader, test_loader = data
    start_time = time.time()
    
    total_words = 0
    print("Starting training.\n")
    batch_size = train_loader.batch_size
    
    for epoch in range(epochs):
        # batch_size = train_loader.batch_size
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
            states = model.detach_states(states)
            scores, states = model(x, states)
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
        print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch+1, valid_perplexity))
        print("*************************************************\n")
    test_perp = get_perplexity(data=test_loader, model=model, batch_size=batch_size)
    print("Test set perplexity : {:.3f}".format(test_perp))
    print("Training is over.")
    return model


def main():
    hyperparams = {
        'embed_dims': None,
        'device': 'cpu', # 'gpu'
        'freq_threshold': 3,
        'dropout_prob': 0.5,
        'init_range': 0.05,
        'epochs': 10,
        'learning_rate': 1,
        'learning_rate_decay': 1.2,
        'num_layers': 2,
        'batch_size': 3, #TODO: on nyt small dataset, we get issues with batch size and timesteps. Not sure why
        'time_steps': 5,
        'max_grad': 5,
        'embed_tying': False,
        'bias': False,
        'save_model': True,
        'load_model': False,
        'model_path': 'lstm_model',
        'topic': 'nyt_covid', # enter 'wiki' or 'nyt_covid'
        'path': 'data/small_test_corpora'
    }

    # set params for init_datasets
    data_params = {k:hyperparams[k] for k in ['topic','freq_threshold', 'time_steps', 'batch_size',  'path']}
    datasets = init_datasets(**data_params)

    # we store the vcab size and word2index dict
    vocab_size = datasets['vocab_size']
    word2index = datasets['word2index']

    # we get the data_loaders: train, valid, test
    data_loaders = datasets['data_loaders']

    # set params for model training
    model_params = ['device', 'embed_dims', 'dropout_prob', 'init_range',
                    'num_layers', 'max_grad', 'embed_tying', 'bias']
    model_params = {k:hyperparams[k] for k in model_params}
    model_params['vocab_size'] = vocab_size
    # Masum recommended this as embed dims
    # TODO: Make this more easily modifiable.
    #   Want to do embed dims = user input if input provied, else embed dims = line below
    model_params['embed_dims'] = int(np.ceil(np.sqrt(np.sqrt(vocab_size))))
    model = LSTM_Model(**model_params)
    print(f"vocab size : {vocab_size}")
    for d, l in zip(data_loaders, ['train', 'valid', 'test']):
        perplexity = get_perplexity(data=d, model=model, batch_size=d.batch_size)
        print("perplexity on %s dataset before training: %.3f, " % (l, perplexity))

    if hyperparams['load_model']:
        model.load_state_dict(torch.load(hyperparams['model_path']))

    else:
        training_params = ['epochs', 'learning_rate', 'learning_rate_decay', 'max_grad']
        training_params = {k:hyperparams[k] for k in training_params}
        model = train(data=data_loaders, model=model, **training_params)

    # now calculate perplexities for train, valid, test
    for d, l in zip(data_loaders, ['train', 'valid', 'test']):
        perplexity = get_perplexity(data=d, model=model, batch_size=d.batch_size)
        print("perplexity on %s dataset after training : %.3f, " % (l, perplexity))

    if hyperparams['save_model']:
        torch.save(model.state_dict(), hyperparams['model_path'])

if __name__=='__main__':
    main()