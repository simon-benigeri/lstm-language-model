import os
from io import open
from typing import List, Dict

PATH='test_corpora'

class Dictionary(object):
    def __init__(self, frequency_threshold):
        self.word2idx = {'<pad_sequences>': 0, '<unk>': 1}
        self.idx2word = ['<pad_sequences>', '<unk>']
        self.word2freq = {}

    def add_word(self, word:str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def filter(self):
        #TODO: apply frequency threshold
        pass

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path:str, topic:str):
        self.dictionary = Dictionary()
        self.train = self.convert_to_int_lists(os.path.join(path, f"{topic}.train.txt"))
        self.valid = self.convert_to_int_lists(os.path.join(path, f"{topic}.valid.txt"))
        self.test = self.convert_to_int_lists(os.path.join(path, f"{topic}.test.txt"))

    def convert_to_int_lists(self, path:str):
        """
        We split the text corpus files into lines,
        lines into tokens,
        and we map those tokens to ints
        :param path:
        :return:
        """
        assert os.path.exists(path)

        # First we populate the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                # we remove have '<s>' because it always follows '</s>'
                line = line.replace('<s>', '')
                # we already have '</s>' to signify end of sentence
                words = line.split()
                # words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content and pad_sequences sequences
        with open(path, 'r', encoding="utf8") as f:
            word_id_sequences = []
            for line in f:
                # we remove have '<s>' because it always follows '</s>'
                line = line.replace('<s>', '').strip()
                # we already have '</s>' to signify end of sentence
                words = line.split()
                # words = line.split() + ['<eos>']
                # word_ids = []
                word_ids = [self.dictionary.word2idx[word] for word in words]
                # for word in words:
                #    word_ids.append(self.dictionary.word2idx[word])
                word_id_sequences.append(word_ids)

        return word_id_sequences
