import os
import time
from io import open
from typing import List, Dict, Tuple
import numpy as np
import nltk

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

    return np.array(train).reshape(-1, 1), np.array(valid).reshape(-1, 1), np.array(test).reshape(-1, 1), len(words)


def minibatches(dataset, batch_size, seq_length):

    return


if __name__=='__main__':
    start_time = time.time()
    # PATH = 'data/test_corpora'
    PATH = 'data/corpora'
    topic = 'nyt_covid'
    train, valid, test, vocab_size = init_corpus(PATH, topic, 3)
    print(train)
    print(f"vocab size = {vocab_size}")
    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))
