from src.datasets import init_datasets
import time
import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple

class LSTM_Model(nn.Module):
    def __init__(self, device:str, vocab_size:int, max_grad:float, embed_dims:int, num_layers:int,
                 dropout_prob:float, init_range:float, bias:bool):
        """
        :param device: cpu, gpu
        :param vocab_size: vocabulary size
        :param max_grad: gradient clipped at this value
        :param embed_dims: output dimensions of embedding layer
        :param num_layers: number of lstm layers
        :param dropout_prob: dropout probability
        :param init_range: weights are initialized by drawing samples
                        from uniform distribution over interval [init_range, init_range]
        :param bias: if true, we use bias weights in nn layers
        """
        super().__init__()
        self.max_grad = max_grad
        self.init_param = init_range

        # model architecture
        self.dropout = nn.Dropout(p=dropout_prob)
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dims)
        lstms = [nn.LSTM(input_size=embed_dims, hidden_size=embed_dims, bias=bias, batch_first=True)
                 for _ in range(num_layers)
                 ]
        self.lstms = nn.ModuleList(modules=lstms)
        self.fc = nn.Linear(in_features=embed_dims, out_features=vocab_size)

        # initialize model weights
        for param in self.parameters():
            nn.init.uniform_(param, -self.init_param, self.init_param)

        self.device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")


    def init_state(self, batch_size:int) -> List[Tuple[Tensor, Tensor]]:
        """
        Initialize weights for hidden states in lstm layers
        :param batch_size: batch size
        :return: initialized weights
        """
        dev = next(self.parameters()).device
        states = [
            (torch.zeros(1, batch_size, layer.hidden_size, device=dev),
             torch.zeros(1, batch_size, layer.hidden_size, device=dev))
                  for layer in self.lstms
        ]
        return states

    def detach_states(self, states:List[Tuple[Tensor, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        """
        Detach output from computational graph
        :param states: lstm weights
        :return: detached view of data
        """
        return [(h.detach(), c.detach()) for (h, c) in states]

    def forward(self, x:Tensor, states:List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        """
        Forward pass through lstm model
        :param x: data with shape (batch_size, time_steps)
        :param states: lstm weights
        :return: (unnormalized token probabilities, lstm weights)
        """
        x = self.embed(x)
        x = self.dropout(x)
        for i, lstm in enumerate(self.lstms):
            x, states[i] = lstm(x, states[i])
            x = self.dropout(x)
        scores = self.fc(x)
        return scores, states
