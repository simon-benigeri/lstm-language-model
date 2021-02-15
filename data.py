import os
import time
from io import open
from typing import List, Dict, Tuple
import numpy as np
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

PATH = 'data/small_test_corpora'

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
    text = text.replace('  ', ' ')
    words = text.strip().split(' ')
    return words


def _apply_freq_threshold(words: List[str], threshold: int) -> List[str]:
    """
    :param words: list of tokens
    :param threshold: frequency threshold
    :return: list of tokens, where tokens with frequency below the threshold are replaced by 'unk'
    """
    # Get Frequency distribution
    freq_dist = nltk.FreqDist(words)
    above_threshold = dict()
    below_threshold = dict()

    #FreqDist.iteritems() returns items in decreasing order of frequency
    for x, f in freq_dist.items():
        if not freq_dist[x] > threshold:
            below_threshold[x] = f
        else:
            above_threshold[x] = f

    filtered_words = ['<unk>' if word in below_threshold else word for word in words]

    return filtered_words


def _init_corpora(path:str, topic:str, freq_threshold:int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """
    :param path:
    :param topic:
    :return:
    """
    # read the text
    train = _load_text_data(os.path.join(path, f"{topic}.train.txt"))
    valid = _load_text_data(os.path.join(path, f"{topic}.valid.txt"))
    test = _load_text_data(os.path.join(path, f"{topic}.test.txt"))

    # split into word/token
    train = _tokenize(text=train)
    valid = _tokenize(text=valid)
    test = _tokenize(text=test)

    # apply frequency threshold to training set
    train = _apply_freq_threshold(words=train, threshold=freq_threshold)
    # create vocabulary: set of words in train and word to index mapping
    words = sorted(set(train))
    words.insert(0, '<pad>')
    word2index = {word: index for index, word in enumerate(words)}

    # convert each word to a list of integers. if word is not in vocab, we use unk
    train = [word2index[word] if word in word2index else word2index['<unk>'] for word in train]
    valid = [word2index[word] if word in word2index else word2index['<unk>'] for word in valid]
    test = [word2index[word] if word in word2index else word2index['<unk>'] for word in test]

    # return list of len n to (n, 1) matrix
    # return np.array(train).reshape(-1, 1), np.array(valid).reshape(-1, 1), np.array(test).reshape(-1, 1), len(words)
    return np.array(train), np.array(valid), np.array(test), word2index


def _generate_io_sequences(data:np.ndarray, time_steps:int) -> Tuple:# -> List[Tuple]:
    """
    :param data: sequence of integer representation of words
    :param time_steps: number of time steps in LSTM cell
    :return:
    """
    data = torch.LongTensor(data)
    # split tensor into tensors of of size time_steps
    data = torch.split(tensor=data, split_size_or_sections=time_steps)

    # note: word2index['<pad>'] = 0
    sequences = pad_sequence(data, batch_first=True, padding_value=0)

    # from seq we generate 2 copies.
    # inputs=seq[:-1], targets=seq[1:]
    sequences_inputs = sequences.narrow_copy(1, 0, sequences.shape[1] - 1)
    sequences_targets = sequences.narrow_copy(1, 1, sequences.shape[1] - 1)

    return (sequences_inputs, sequences_targets)

def _generate_io_sequences_2(data:np.ndarray, time_steps:int) -> Tuple[np.ndarray, np.ndarray]:
    inputs = data
    targets = data[1:] + [0]

    for index in range(len(inputs)):
        pass

    return inputs, targets


class Sequence_Data(Dataset):
    def __init__(self, x:torch.LongTensor, y:torch.LongTensor):
        self.x = x
        self.y = y
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


def _intlist_to_dataloader(data:np.ndarray, time_steps:int, batch_size:int) -> DataLoader:
    """
    :param data: input list of integers
    :param batch_size: hyper parameter, for minibatch size
    :param time_steps: hyper parameter for sequence length for bptt
    :return: DataLoader for SGD
    """
    # given int list, generate input and output sequences of length = time_steps
    inputs, targets = _generate_io_sequences(data=data, time_steps=time_steps)

    # create Dataset object
    dataset = Sequence_Data(x=inputs, y=targets)

    # create dataloader
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return data_loader


def init_datasets(topic:str, freq_threshold:int, time_steps:int, batch_size:int, path:str=PATH) -> Dict:
    """
    :param path: path to data files: [topic].train.txt
    :param topic: [topic].train.txt, where topic can be wikitext, or nyt_covid
    :param freq_threshold: hyperparam, words in training set with freq < threshold are replaced by '<unk>'
    :param time_steps: hyperparam, number of time steps and therefore seq_length for bptt
    :param batch_size: hyperparam, batch size
    :return: datasets dict
    """
    train, valid, test, word2index = _init_corpora(path=path, topic=topic, freq_threshold=freq_threshold)
    train_loader = _intlist_to_dataloader(data=train, time_steps=time_steps, batch_size=batch_size)
    valid_loader = _intlist_to_dataloader(data=valid, time_steps=time_steps, batch_size=batch_size)
    test_loader = _intlist_to_dataloader(data=test, time_steps=time_steps, batch_size=batch_size)
    datasets = {
        'data_loaders': (train_loader, valid_loader, test_loader),
        'word2index': word2index,
        'vocab_size': len(word2index)
    }
    return datasets


if __name__=='__main__':
    start_time = time.time()
    # PATH = 'data/test_corpora'
    path = 'data/small_test_corpora'
    topic = 'nyt_covid'

    data = np.array(list(range(1,11)))
    seqs = _generate_io_sequences(data, time_steps=5)
    print(seqs)


    """
    # start training loop
    for epoch in range(epochs):
        for step, (x, y) in enumerate(test_loader):  # gives batch data
            print(x, y)
            # print(ass)
        pass
    """

    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))
