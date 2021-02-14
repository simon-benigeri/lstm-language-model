from lstm import Model
from data import get_data
import torch
import torch.nn as nn
import torch.nn.utils.clip_grad_norm_ as grad_clip
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


def get_perplexity(model, dataset):
    pass


def train(model, data, epochs, learning_rate, learning_rate_decay, max_grad):
    train_data, valid_data, test_data = data.values()
    start_time = time.time()

    print("Starting training.\n")

    for epoch in range(epochs):
        model.train()

        if epoch > 5:
            learning_rate = learning_rate / learning_rate_decay

        for i, (x, y) in enumerate(train_data):
            batch_size = len(x)
            model.zero_grad()
            states = model.detach(states)
            scores, states = model(x, states)
            loss = nll_loss(scores, y)
            loss.backward()
            with torch.no_grad():
                norm = grad_clip(model.parameters(), max_grad)
                for param in model.parameters():
                    param -= learning_rate * param.grad
            if i % (len(train_data) // 10) == 0:
                end_time = time.time()
                print("batch no = {:d} / {:d}, ".format(i, len(train_data)) +
                      "train loss = {:.3f}, ".format(loss.item() / batch_size) +
                      "dw.norm() = {:.3f}, ".format(norm) +
                      "lr = {:.3f}, ".format(learning_rate) +
                      "since beginning = {:d} mins, ".format(round((end_time-start_time)/60)) +
                      "cuda memory = {:.3f} GBs".format(torch.cuda.max_memory_allocated()/1024/1024/1024))
        model.eval()
        valid_perplexity = get_perplexity(model, valid_data)
        print("Epoch : {:d} || Validation set perplexity : {:.3f}".format(epoch+1, valid_perplexity))
        print("*************************************************\n")
    test_perp = get_perplexity(model, test_data)
    print("Test set perplexity : {:.3f}".format(test_perp))
    print("Training is over.")

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
        'timesteps': 35,
        'max_grad': 5,
        'embed_tying': False,
        'bias': False,
        'save_model': True,
        'load_model': True,
        'model_path': 'lstm_model'
    }

    data_params = {k:hyperparams for k in ['freq_threshold', 'timesteps', 'batch_size']}
    datasets = get_data(**data_params)

    model_params = ['embed_dims', 'dropout_prob', 'init_range', 'num_layers', 'step_size', 'max_grad', 'embed_tying', 'bias']
    model_params = {k:hyperparams[k] for k in model_params}
    model = Model(**model_params)

    if hyperparams['load_model']:
        model.load_state_dict(torch.load(hyperparams['model_path']))

    else:
        training_params = ['epochs', 'learning_rate', 'learning_rate_decay', 'max_grad']
        training_params = {k:hyperparams[k] for k in training_params}
        perplexities, model = train(model, datasets['data_loaders'], **training_params)

    if hyperparams['save_model']:
        torch.save(model.state_dict(), hyperparams['model_path'])