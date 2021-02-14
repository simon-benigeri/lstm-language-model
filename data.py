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
    return np.array(train), np.array(valid), np.array(test), len(words)

def generate_sequences_of_size_timestep(data, timestep_size):
    # TODO: IMPLEMENT
    pass

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


def minibatch(data, batch_size, timestep_size):
    """

    :param data: n, 1 matrix
    :param batch_size: how many samples per batch
    :param timestep_size: timestep length
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
    dataset = []
    # for i in range 0, num_batches - 1, increment by seq_length
    for i in range(0, data.size(1)-1, timestep_size):
        sequence_length = int(np.min([timestep_size, data.size(1)-1-i]))

        if sequence_length <data.size(1)-1-i:

            x = data[:,i:i+sequence_length ].transpose(1, 0)
            y = data[:,i+1:i+sequence_length +1].transpose(1, 0)
            dataset.append((x, y))
    print(dataset)
    return dataset


if __name__=='__main__':
    start_time = time.time()
    # PATH = 'data/test_corpora'
    PATH = 'data/test_corpora'
    topic = 'nyt_covid'
    # train, valid, test, vocab_size = init_corpus(PATH, topic, 3)
    # print(test)
    k = (list(range(41)))
    a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]).reshape(-1, 1)
    b = np.array(list(range(41))).reshape(-1, 1)
    # test_batches = minibatch(a, batch_size=3, timestep_size=2)
    test_batches = minibatch(b, batch_size=20, timestep_size=5)
    print(test_batches)
    """
    
    print(f"vocab size = {vocab_size}")
    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))
    """