import os
from io import open
from typing import List, Dict
import numpy as np

PATH='test_corpora'

def _load_text_data(path: str) -> List[str]:
    with open(path, 'r') as f:
        text = f.read() #f.readlines()
    text = text.replace('<s>', '')
    text = text.replace('\n', '')
    text = text.strip()
    return text.split(' ')

def apply_frequency_threshold(word_index_list):
    above_p = []
    for (x, f) in FreqDist.iteritems():
        if not f > p:
            break
        above_p.append((x, f))

def init_corpus(path:str, topic:str):
    train = _load_text_data(os.path.join(path, f"{topic}.train.txt"))
    valid = _load_text_data(os.path.join(path, f"{topic}.valid.txt"))
    test = _load_text_data(os.path.join(path, f"{topic}.test.txt"))
    words = sorted(set(train))
    word2index = {word: index for index, word in enumerate(words)}
    train = [word2index[c] for c in train]
    valid = [word2index[c] for c in valid]
    test = [word2index[c] for c in test]
    return np.array(train).reshape(-1, 1), np.array(valid).reshape(-1, 1), np.array(test).reshape(-1, 1), len(words)

if __name__=='__main__':
    PATH = 'test_corpora'
    topic = 'nyt_covid'
    train = _load_text_data(os.path.join(PATH, f"{topic}.train.txt"))
    print(train)

    train, valid, test, vocab_size = init_corpus(PATH, topic)
