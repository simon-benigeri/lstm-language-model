from lstm import Model


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

def main():
    hyperparams = {
        'embed_dims': None, #
        'freq_threshold': 3,
        'dropout_prob': 0.5, #
        'init_range': 0.05, #
        'epochs': 40,
        'learning_rate': 1,
        'learning_rate_decay': 1.2,
        'num_layers': 2, #
        'batch_size': 20,
        'step_size': 35,
        'max_grad': 5,          # ???
        'embed_tying': False,   # ???
        'bias': False           # ???
    }



    train_data, valid_data, test_data = get_data(threshold=hyperparams['freq_threshold'])

    model_params = ['embed_dims', 'dropout_prob', 'init_range', 'num_layers', 'step_size', 'max_grad', 'embed_tying', 'bias']
    model_params = {k:hyperparams[k] for k in model_params}
    model = Model(**model_params)



def get_data():
    return None, None, None