import os
import time
from io import open
from typing import List, Dict, Tuple
import numpy as np
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

class Sequence_Dataset(Dataset):
    def __init__(self, x:torch.LongTensor, y:torch.LongTensor):
        self.x = x
        self.y = y
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

def _load_text_data(path: str) -> str:
    """
    read text file
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        text = f.read() #f.readlines()
    return text


def _tokenize(text: str) -> List[str]:
    """
    Remove unwanted tokens like <s> and \n
    :param text: tokens separated by ' '
    :return: list of tokens
    """
    text = text.replace('<s>', '')
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())
    text = text.lower()
    words = text.strip().split(' ')
    return words

def _init_corpora(path:str, topic:str, freq_threshold:int,) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    :param path:
    :param topic:
    :return:
    """
    # read the text
    train = _load_text_data(os.path.join(path, f"{topic}.train.txt"))
    valid = _load_text_data(os.path.join(path, f"{topic}.valid.txt"))
    test = _load_text_data(os.path.join(path, f"{topic}.test.txt"))

    if topic == 'wiki':
        # remove wiki tags
        train = ' '.join(train.replace('=', ' ').split())
        valid = ' '.join(valid.replace('=', ' ').split())
        test = ' '.join(test.replace('=', ' ').split())

    # split into word/token
    train = _tokenize(text=train)
    valid = _tokenize(text=valid)
    test = _tokenize(text=test)

    # apply frequency threshold to training set
    freq_dist = nltk.FreqDist(train)
    train = ['<unk>' if freq_dist[t] < freq_threshold else t for t in train]

    # create vocabulary: set of words in train and word to index mapping
    vocab = sorted(set(train))
    word2index = {word: index+1 for index, word in enumerate(vocab)}
    word2index['<pad>'] = 0

    # convert each word to a list of integers. if word is not in vocab, we use unk
    train = [word2index[word] for word in train]
    valid = [word2index[word] if word in word2index else word2index['<unk>'] for word in valid]
    test = [word2index[word] if word in word2index else word2index['<unk>'] for word in test]

    # return (n, ) arrays for train, valid, test, and the word2index dict
    return np.array(train), np.array(valid), np.array(test), word2index

def _generate_io_sequences(sequence: np.ndarray, time_steps: int) -> Tuple:
    """
    :param sequence: sequence of integer representation of words
    :param time_steps: number of time steps in LSTM cell
    :return: Tuple of torch tensors of shape (n, time_steps)
    """
    sequence = torch.LongTensor(sequence)

    # from seq we generate 2 copies.
    inputs, targets = sequence, sequence[1:]

    # split seq into seq of of size time_steps
    inputs = torch.split(tensor=inputs, split_size_or_sections=time_steps)
    targets = torch.split(tensor=targets, split_size_or_sections=time_steps)

    # recall: word2index['<pad>'] = 0
    inputs_padded = pad_sequence(sequences=inputs, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(sequences=targets, batch_first=True, padding_value=0)

    return (inputs_padded, targets_padded)


def _build_dataloader(data:np.ndarray, time_steps:int, batch_size:int) -> DataLoader:
    """
    :param data: input list of integers
    :param batch_size: hyper parameter, for minibatch size
    :param time_steps: hyper parameter for sequence length for bptt
    :return: DataLoader for SGD
    """
    # given int list, generate input and output sequences of length = time_steps
    inputs, targets = _generate_io_sequences(sequence=data, time_steps=time_steps)
    
    # cut off any data that will create incomplete batches
    num_batches = len(inputs) // batch_size
    inputs = inputs[:num_batches*batch_size]
    targets = targets[:num_batches*batch_size]
    
    # create Dataset object and from it create data loader
    dataset = Sequence_Dataset(x=inputs, y=targets)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def init_datasets(topic:str, freq_threshold:int, time_steps:int, batch_size:int, path:str) -> Dict:
    """
    :param path: path to data files: [topic].train.txt
    :param topic: [topic].train.txt, where topic can be wikitext, or nyt_covid
    :param freq_threshold: hyperparam, words in training set with freq < threshold are replaced by '<unk>'
    :param time_steps: hyperparam, number of time steps and therefore seq_length for bptt
    :param batch_size: hyperparam, batch size
    :return: datasets dict
    """
    train, valid, test, word2index = _init_corpora(path=path, topic=topic, freq_threshold=freq_threshold)
    train_loader = _build_dataloader(data=train, time_steps=time_steps, batch_size=batch_size)
    valid_loader = _build_dataloader(data=valid, time_steps=time_steps, batch_size=batch_size)
    test_loader = _build_dataloader(data=test, time_steps=time_steps, batch_size=batch_size)
    datasets = {
        'data_loaders': (train_loader, valid_loader, test_loader),
        'word2index': word2index,
        'vocab_size': len(word2index)
    }
    return datasets
