import os
import time
from io import open
from typing import List, Dict, Tuple
import numpy as np
import nltk
import torch

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
    word2index = {word: index for index, word in enumerate(words)}

    # convert each word to a list of integers. if word is not in vocab, we use unk
    train = [word2index[word] if word in word2index else word2index['<unk>'] for word in train]
    valid = [word2index[word] if word in word2index else word2index['<unk>'] for word in valid]
    test = [word2index[word] if word in word2index else word2index['<unk>'] for word in test]

    # return list of len n to (n, 1) matrix
    return np.array(train).reshape(-1, 1), np.array(valid).reshape(-1, 1), np.array(test).reshape(-1, 1), len(words)


def minibatch(data, batch_size, seq_length):
    """

    :param data: n, 1 matrix
    :param batch_size: how many samples per batch
    :param seq_length: timestep length
    :return:
    """
    # (n, 1) tensor
    data = torch.tensor(data, dtype=torch.int64)
    # number of batches m is n // batch_size
    num_batches = data.size(0) // batch_size
    # cut off data that doesn't fit into equally sized batches
    data = data[:num_batches * batch_size]
    # reshape into (m, n)
    data = data.view(batch_size, -1)

    # for i in range 0, num_batches - 1, increment by seq_length
    for i in range(0, data.size(1)-1, seq_length):

        seqlen=int(np.min([seq_length,data.size(1)-1-i]))

        if seqlen<data.size(1)-1-i:
            x=data[:,i:i+seqlen].transpose(1, 0)
            y=data[:,i+1:i+seqlen+1].transpose(1, 0)
            dataset.append((x, y))

    return dataset


if __name__=='__main__':
    start_time = time.time()
    # PATH = 'data/test_corpora'
    PATH = 'data/test_corpora'
    topic = 'nyt_covid'
    train, valid, test, vocab_size = init_corpus(PATH, topic, 3)
    print(test)
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(-1, 1)
    test_batches = minibatch(a, batch_size=2, seq_length=3)
    print(test_batches)
    print(f"vocab size = {vocab_size}")
    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))
