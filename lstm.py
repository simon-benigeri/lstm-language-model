from data import init_datasets
import torch
import torch.nn as nn
import math

class LSTM_Model(nn.Module):
    def __init__(self, vocab_size:int, max_grad:float, embed_dims:int, num_layers:int,
                 dropout_prob:float, init_param:float, bias:bool, embed_tying:bool):
        """
        :param vocab_size:
        :param max_grad:
        :param embed_dims:
        :param num_layers:
        :param dropout_prob:
        :param init_param:
        :param bias:
        :param embed_tying:
        """
        super().__init__()
        self.max_grad = max_grad
        self.init_param = init_param
        self.bias = bias
        self.embed_tying = embed_tying

        # model architecture
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=dropout_prob)
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dims)
        self.lstm_modules = [nn.LSTM(input_size=embed_dims, hidden_size=embed_dims, bias=bias)
                             for _ in range(num_layers)
                             ]
        self.lstm_modules = nn.ModuleList(modules=self.lstm_modules)
        self.fc = nn.Linear(in_features=embed_dims, out_features=vocab_size, bias=bias)

        self._reset_parameters()

    # set intial parameters
    def _reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.init_param, self.init_param)

    def _init_state(self, batch_size):
        dev = next(self.parameters()).device
        states = [
            (torch.zeros(1, batch_size, layer.hidden_size, device=dev),
             torch.zeros(1, batch_size, layer.hidden_size, device=dev))
                  for layer in self.lstm_modules
        ]
        return states

    def _detach(self, states):
        return [(h._detach(), c._detach()) for (h, c) in states]

    def _forward(self, x, states):
        x = self.embed(x)
        x = self.dropout(x)
        for i, lstm in enumerate(self.lstm_modules):
            x, states[i] = lstm(x, states[i])
            x = self.dropout(x)
        scores = self.fc(x)
        return scores, states

if __name__ == "__main__":
    datasets = init_datasets(topic='nyt_covid', freq_threshold=2, time_steps=35, batch_size=20)
    data_loaders = datasets['data_loaders']
    vocab_size = datasets['vocab_size']
    hidden_size = math.ceil(math.sqrt(math.sqrt(datasets['vocab_size'])))
    model = LSTM_Model(vocab_size=1000, max_grad=5, embed_dims=200, num_layers=2,
                       dropout_prob=0.5, init_param=0.05, bias=False, embed_tying=False)
    print(model._modules)