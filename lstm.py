import os
from data import Corpus
from typing import List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

PATH='test_corpora'

class LM_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

net = LM_LSTM()
print(net)

if __name__ == "__main__":
    PATH = 'test_corpora'
    topic = 'nyt_covid' # 'wiki'
    corpus = Corpus(path=PATH, topic=topic)
    print(corpus.test)
