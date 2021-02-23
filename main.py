from models.lstm import LSTM_Model
from pipelines.datasets import init_datasets
import torch
import argparse

from utils.statistics import perplexity
from pipelines.trainer import train

parser = argparse.ArgumentParser(description="Language model utilizing LSTM of Zeremba et al. (2014).")
parser.add_argument("--embed_dims", type=int, default=200, help="Output dimensions of embedding layer.")
parser.add_argument("--dropout_prob", type=float, default=0.5, help = "The probability of dropout being applied to a cell.")
parser.add_argument("--init_range", type=float, default=0.05, help = "Parameters will be initialized between +/- this value.")
parser.add_argument("--num_layers", type=int, default=2, help = "Number of layers in LSTM.")
parser.add_argument("--time_steps", type=int, default=30, help = "Length of one sequence/number of tokens in a sequence.")
parser.add_argument("--embed_tying", type=bool, default=False, help = "Whether input and output weights are the same during training.")
parser.add_argument("--bias", type=bool, default=False, help = "Bias applied to fully-connected layer.")
parser.add_argument("--freq_threshold", type=int, default=3, help="Minimum frequency threshold which will be applied to the corpus.")
parser.add_argument("--epochs", type=int, default=20, help = "Number of epochs the model is trained for.")
parser.add_argument("--learning_rate", type=float, default=1.0, help = "The learning rate applied during training.")
parser.add_argument("--learning_rate_decay", type=float, default=1.25, help = "Factor by which learning rate is decreased.")
parser.add_argument("--batch_size", type=int, default=20, help = "The batch size used during training.")
parser.add_argument("--max_grad", type=int, default=2, help = "Maximum gradient in training (higher gradient values clipped).")
parser.add_argument("--device", type=str, choices=["cpu", "gpu", "tpu"], default="gpu", help = "Whether to use cpu, gpu or tpu.")
parser.add_argument("--save_model", type=bool, default=True, help = "Whether model will be saved after training.")
parser.add_argument("--load_model", type=bool, default=False, help = "Whether model will be loaded from file (no training will occur).")
parser.add_argument("--model_path", type=str, default="saved/lstm_model", help = "The path at which to save the model.")
parser.add_argument("--topic", type=str, choices=["wiki", "nyt_covid"], default="wiki", help = "The topic which will be loaded.")
parser.add_argument("--path", type=str, default="data/", help = "The path to the cleaned corpus files.")
args = vars(parser.parse_args())


def main(args):
    data_params = {k:args[k] for k in ['topic','freq_threshold', 'time_steps', 'batch_size',  'path']}
    datasets = init_datasets(**data_params)

    # data_loaders are (train, valid, test)
    data_loaders = datasets['data_loaders']

    model_params = ['device', 'embed_dims', 'dropout_prob', 'init_range', 'num_layers', 'max_grad', 'bias']
    model_params = {k:args[k] for k in model_params}
    model_params['vocab_size'] = datasets['vocab_size']

    model = LSTM_Model(**model_params)

    # print out vocab size and perplexity on validation set
    print(f"vocab size : {datasets['vocab_size']}")
    px = perplexity(data=data_loaders[1], model=model, batch_size=data_loaders[1].batch_size)
    print("perplexity on %s dataset before training: %.3f, " % ('valid', px))

    if args.load_model:
        model.load_state_dict(torch.load(args.model_path))
    else:
        training_params = ['epochs', 'learning_rate', 'learning_rate_decay', 'max_grad']
        training_params = {k:args[k] for k in training_params}
        model, perplexity_scores = train(data=data_loaders, model=model, **training_params)

    # now calculate perplexities for train, valid, test
    for d, name in zip(data_loaders, ['train', 'valid', 'test']):
        px = perplexity(data=d, model=model, batch_size=d.batch_size)
        print("perplexity on %s dataset after training : %.3f, " % (name, px))

    # save model
    if args.save_model:
        torch.save(model.state_dict(), args.model_path)

if __name__=='__main__':
    main(args)