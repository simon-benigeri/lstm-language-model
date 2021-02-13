from data import Corpus
from typing import List, Dict
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def pad_sequences(word_id_sequences):
    tensors = [torch.tensor(word_id).type(torch.int64) for word_id in word_id_sequences]
    # we pad_sequences these sequences
    padded_tensors = pad_sequence(tensors, batch_first=True, padding_value=0)
    return padded_tensors

def generate_inputs_and_targets(padded_tensors):
    input_sequences = padded_tensors.narrow_copy(1, 0, padded_tensors.shape[1] - 1)
    target_sequences = padded_tensors.narrow_copy(1, 1, padded_tensors.shape[1] - 1)
    return input_sequences, target_sequences

def create_seq_to_seq_data(sequences):
    return generate_inputs_and_targets(pad_sequences(sequences))


if __name__=='__main__':
    PATH = 'test_corpora'
    topic = 'nyt_covid'  # 'wiki'
    corpus = Corpus(path=PATH, topic=topic)

    # train_X, train_y = create_seq_to_seq_data(corpus.train)
    train_X, train_y = create_seq_to_seq_data(corpus.valid)
    dict_size = len(corpus.dictionary)
    seq_len = len(train_X[0])
    batch_size = 5
    print(train_X)
    print(train_y)





