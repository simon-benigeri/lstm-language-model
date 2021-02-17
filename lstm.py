from data import init_datasets
import time
import numpy as np
import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, device:str, vocab_size:int, max_grad:float, embed_dims:int, num_layers:int,
                 dropout_prob:float, init_range:float, bias:bool, embed_tying:bool):
        """
        :param vocab_size:
        :param max_grad:
        :param embed_dims:
        :param num_layers:
        :param dropout_prob:
        :param init_range:
        :param bias:
        :param embed_tying:
        """
        super().__init__()
        self.max_grad = max_grad
        self.init_param = init_range
        self.bias = bias
        self.embed_tying = embed_tying
        # model architecture
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(p=dropout_prob)
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dims)
        lstms = [nn.LSTM(input_size=embed_dims, hidden_size=embed_dims, bias=bias, batch_first=True)
                 for _ in range(num_layers)
                 ]
        self.lstms = nn.ModuleList(modules=lstms)
        self.fc = nn.Linear(in_features=embed_dims, out_features=vocab_size, bias=bias)

        self.reset_parameters()

        if device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    # set intial parameters
    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -self.init_param, self.init_param)

    def init_state(self, batch_size):
        dev = next(self.parameters()).device

        states = [
            (torch.zeros(1, batch_size, layer.hidden_size, device=dev),
             torch.zeros(1, batch_size, layer.hidden_size, device=dev))
                  for layer in self.lstms
        ]
        
        return states

    def detach_states(self, states):
        return [(h.detach(), c.detach()) for (h, c) in states]

    def forward(self, x, states):
        x = self.embed(x)
        x = self.dropout(x)
        for i, lstm in enumerate(self.lstms):
            x, states[i] = lstm(x, states[i])
            x = self.dropout(x)
        scores = self.fc(x)
        return scores, states

if __name__ == "__main__":
    start_time = time.time()
    batch_size = 20
    time_steps = 10
    freq_threshold = 1
    epochs = 2

    datasets = init_datasets(topic='wiki', freq_threshold=freq_threshold, time_steps=time_steps,
                             batch_size=batch_size, path='data/corpora')

    data_loaders = datasets['data_loaders']
    vocab_size = datasets['vocab_size']
    embed_dims = int(np.ceil(np.sqrt(np.sqrt(vocab_size))))
    model = LSTM_Model(vocab_size=vocab_size, max_grad=5, embed_dims=embed_dims, num_layers=2,
                       dropout_prob=0.5, init_range=0.05, bias=False, embed_tying=False)

    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))