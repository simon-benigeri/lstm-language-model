import os
import time
from io import open
from typing import List, Dict, Tuple
import numpy as np
import nltk
import torch
from torch.nn.utils.rnn import pad_sequence

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


def apply_freq_threshold(words: List[str], threshold: int) -> List[str]:
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


def init_corpus(path:str, topic:str, frequency_threshold:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
    train = apply_freq_threshold(words=train, threshold=frequency_threshold)
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
    return np.array(train), np.array(valid), np.array(test), len(words)


def generate_datasets(data:np.ndarray, time_steps:int) -> List[Tuple]:
    """
    :param data: sequence of integer representation of words
    :param time_steps: number of time steps in LSTM cell
    :return:
    """
    data = torch.tensor(data, dtype=torch.int64)
    # split tensor into tensors of of size time_steps
    data = torch.split(tensor=data, split_size_or_sections=time_steps)

    # note: word2index['<pad>'] = 0
    sequences = pad_sequence(data, batch_first=True, padding_value=0)

    # from seq we generate 2 copies.
    # inputs=seq[:-1], targets=seq[1:]
    sequences_input = sequences.narrow_copy(1, 0, sequences.shape[1] - 1)
    sequences_target = sequences.narrow_copy(1, 1, sequences.shape[1] - 1)

    # dataset is a list of tuples to pair each input with its respective target
    dataset = [(X, y) for X, y in zip(sequences_input, sequences_target)]

    return dataset


if __name__=='__main__':
    start_time = time.time()
    # PATH = 'data/test_corpora'
    PATH = 'data/test_corpora'
    topic = 'nyt_covid'
    # train, valid, test, vocab_size = init_corpus(PATH, topic, 3)
    a = np.array(list(range(21)))
    dataset = generate_datasets(data=a, time_steps=5)
    # print(dataset)
    """
    
    print(f"vocab size = {vocab_size}")
    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))
    """