from data import init_datasets
import torch
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, layer_num, dropout, init_param):
        """
        Initialization for the model.
        Args:
            vocab_size: Nuber of unique tokens in the dataset
            hidden_size: Embedding dimensions. (the 4th root thing)
            layer_num: Number of lstm layers. For stacked LSTM
            init_param: setting initial parameters
            dropout: probability for dropping a node in network
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.init_param = init_param
        self.dropout = nn.Dropout(p=dropout)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnns = [nn.LSTM(hidden_size, hidden_size) for i in range(layer_num)]
        self.rnns = nn.ModuleList(self.rnns)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.init_param, self.init_param)

    def state_init(self, batch_size):
        dev = next(self.parameters()).device
        states = [
            (torch.zeros(1, batch_size, layer.hidden_size, device=dev),
             torch.zeros(1, batch_size, layer.hidden_size, device=dev))
                  for layer in self.rnns
        ]
        return states

    def detach(self, states):
        return [(h.detach(), c.detach()) for (h, c) in states]

    def forward(self, x, states):
        x = self.embed(x)
        x = self.dropout(x)
        for i, rnn in enumerate(self.rnns):
            x, states[i] = rnn(x, states[i])
            x = self.dropout(x)
        scores = self.fc(x)
        return scores, states

if __name__ == "__main__":
    print('hello darkness my old friend')
    datasets = init_datasets(topic='nyt_covid', freq_threshold=0, time_steps=5, batch_size=10)
    data_loaders = datasets['data_loaders']
    vocab_size = datasets['vocab_size']
    hidden_size = math.ceil(math.sqrt(math.sqrt(datasets['vocab_size'])))
    model = Model(vocab_size, hidden_size, layer_num=2, dropout=0.1, init_param=0.1)
    print(model.__dict__)