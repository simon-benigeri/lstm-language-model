import os
import re
import time
import json
from typing import List, Dict
from nltk.corpus import stopwords

METADATA_TAGS = {
    '_titel_': '<<start_title>>',               # Start-of-title token
    '_/titel_': '<<end_title>>',                # End-of-title token
    '_pub_time_': '<<start_pub_time>>',         # start-of-publication-time token
    '_/pub_time_': '<<end_pub_time>>',          # start-of-publication-time token
    '_articel_id_': '<<start_article_id>>',     # start-of-article-id token
    '_/articel_id_': '<<end_article_id>>',      # end-of-article-id token
    '_despriction_': '<<start_desc>>',          # start-of-description token
    '_/despriction_': '<<end_desc>>',           # end-of-description token
    '_news_keywords_': '<<start_keywords>>',    # start-of-keywords token
    '_/news_keywords_': '<<end_keywords>>',     # end-of-keywords token
    '_sect_': '<<start_section>>',              # start-of-section token
    '_/sect_': '<<end_section>>',               # end-of-section token
    '_s_': '<s>',                               # start-of-sentence token
    '_/s_': '</s>',                             # end-of-sentence token
    '_start_article_': '<<start_article>>',     # Start-of-article token
    '_end_article_': '<<end_article>>',         # End-of-article token
    '_start_article_body_': '<<start_body>>',   # Start-of-article-body token
    '_end_article_body_': '<<end_body>>'        # End-of-article-body token
    # '_end_article_body': '<<end_body>>'        # Badly spelt End-of-article-body token
}

NON_WORD_TOKENS = {
        # 'SOA': '<<start_article>>',           # Start-of-article token
        # 'EOA': '<<end_article>>',             # End-of-article token
        'SOS': '<s>',                           # Start-of-sentence token
        'EOS': '</s>',                          # End-of-sentence token
        'UNK': '<unk>'                          # Unknown token
}

def _read_text(path: str) -> str:
    with open(path, 'r') as f:
        text = f.read() #f.readlines()
    return text


def load_data(path: str) -> Dict:
    dataset = {
        'train': _read_text(os.path.join(path, 'train.txt')),
        'valid': _read_text(os.path.join(path, 'valid.txt')),
        'test': _read_text(os.path.join(path, 'test.txt'))
    }
    return dataset


def _substitute_tags(text:str, dictionary:Dict) -> str:
    # sort keys by length, in reverse order
    for item in sorted(dictionary.keys(), key=len, reverse=True):
        text = re.sub(item, dictionary[item], text)
    return text


def _select_content_between_tags(text:str, start_tag:str, end_tag:str) -> str:
    pattern = fr"(?<={start_tag}).*?(?={end_tag})"
    texts = re.findall(pattern, text)
    text = ' '.join(texts).strip()
    return text


def _clean_text(text: str) -> str:
    text = re.sub(pattern=r'\n', repl=' ', string=text)
    text = re.sub(pattern=r'\\n', repl=' ', string=text)
    text = re.sub(pattern=r'  ', repl=' ', string=text)
    # text = re.sub(pattern=fr" +{NON_WORD_TOKENS['SOS']}", repl=NON_WORD_TOKENS['SOS'], string=text)
    text = _substitute_tags(text=text, dictionary=METADATA_TAGS)
    text = _select_content_between_tags(text=text, start_tag='<<start_body>>', end_tag='<<end_body>>')
    text = text.lower().strip()
    return text


def clean_dataset(dataset: Dict) -> Dict:
    cleaned_dataset = {
        corpus: _clean_text(text) for corpus, text in dataset.items()
    }
    return cleaned_dataset


def write_to_file(text: str, name:str, path: str):
    with open(os.path.join(path, name), 'w') as f:
        f.write(text)
    return


def write_to_json(object: Dict, name:str, path:str):
    with open(os.path.join(path, name), 'w') as f:
        f.write(json.dumps(object, indent=2))
    return

class Vocabulary:
    def __init__(self, dataset:Dict=None, stop_words:set=None, frequency_threshold:int=0):
        ## INPUTS
        # data
        self.dataset = dataset
        self.stopwords = stop_words
        # hyper params
        self.threshold = frequency_threshold
        # initialize some variables
        self.vocab_size = 0
        self.num_word = 1
        ## OUTPUTS
        # initial vocab
        self.vocab_initial = {NON_WORD_TOKENS['UNK']: (0, 0)}
        # vocab after threshold
        self.vocab = {NON_WORD_TOKENS['UNK']: (0, 0)}
        self.lexicon = ''
        # integer representations
        self.train = []
        self.test = []
        self.valid = []
        # text given integer representation
        self.corpus_train = ''
        self.corpus_valid = ''
        self.corpus_test = ''

        # we save this output to file in case it saves time for computing stats
        self.vocab_output_json = {}

    def _read_training_data(self, corpus: str):
        """
        :param corpus:
        :return:
        """
        # split corpus by space
        for word in corpus.split(' '):
            self._add_word(word)

    def _add_word(self, word: str):
        if word not in self.vocab_initial:
            # First entry of word into vocabulary (index, count)
            self.vocab_initial[word] = (self.num_word, 1)
            self.num_word += 1
        else:
            # Word exists; increase word count
            # tup is (index, count)
            tup = self.vocab_initial[word]
            self.vocab_initial[word] = (tup[0], tup[1] + 1)

    def _apply_frequency_threshold(self):
        for word in self.vocab_initial:
            tup = self.vocab_initial[word]
            # if word count less than frequency threshold
            # recall tup is (index, count)
            if tup[1] <= self.threshold:
                # increase word count for <unk> by that word's count
                unk_tup = self.vocab[NON_WORD_TOKENS['UNK']]
                self.vocab[NON_WORD_TOKENS['UNK']] = (unk_tup[0], unk_tup[1] + tup[1])
            else:
                self.vocab[word] = (len(self.vocab), tup[1])

    def _words_to_ints(self, text: str) -> List[int]:
        # for each word in text, write its index in the vocabulary instead
        return [
            self.vocab[word][0] if word in self.vocab
            else self.vocab[NON_WORD_TOKENS['UNK']][0]
            for word in text.split(' ')
        ]

    def create_vocabulary(self):
        # reading the articles in train we create a vocab dict: { word: (index, count) }
        # read articles and words in each article in training set
        self._read_training_data(corpus=self.dataset['train'])
        # apply frequency threshold on vocab dict
        self._apply_frequency_threshold()
        # using the vocab dict { word: (index in training set, count) }
        # map words in each corpus to integer index of corresponding word in vocab
        self.train = self._words_to_ints(text=self.dataset['train'])
        self.valid = self._words_to_ints(text=self.dataset['valid'])
        self.test = self._words_to_ints(text=self.dataset['test'])
        self.vocab_size = len(self.vocab)
        self.vocab_output_json = {
            'integer_representations': {
                'train': self.train,
                'valid': self.valid,
                'test': self.test
            },
            'vocabs': {
                'vocab': {
                    'data': self.vocab,
                    'size': len(self.vocab)
                },
                'initial_vocab': {
                    'data': self.vocab_initial,
                    'size': len(self.vocab_initial)
                }
            },
            'stopwords': list(self.stopwords),
            'threshold': self.threshold
        }

    def _create_corpus(self, words_as_ints:List[int]) -> str:
        word_list = [list(self.vocab.keys())[index] for index in words_as_ints]
        corpus = self._format_corpus(words=word_list)
        return corpus

    def _format_corpus(self, words: List[str]) -> str:
        text = (' ').join(words)
        text = text.replace(NON_WORD_TOKENS['EOS'], NON_WORD_TOKENS['EOS']+'\n')
        lines = [line.strip() for line in text.splitlines()]
        text = ('\n').join(lines)
        return text

    def create_corpora(self):
        self.corpus_train = self._create_corpus(words_as_ints=self.train)
        self.corpus_valid = self._create_corpus(words_as_ints=self.valid)
        self.corpus_test = self._create_corpus(words_as_ints=self.test)

    def compute_stats(self, corpus):
        pass

if __name__ == "__main__":
    start_time = time.time()

    # set paths and hyper param
    INPUT_PATH = '../data/raw_text_files'  # '../data/nyt_small_samples' #
    OUTPUT_CORPUS_PATH = '../data/corpora'
    OUTPUT_VOCAB_PATH = '../data/output_vocab'
    OUTPUT_LEXICONS_PATH = '../data/lexicons'

    FREQUENCY_THRESHOLD = 3

    # load datasets
    dataset = load_data(path=INPUT_PATH)

    # clean dataset
    cleaned_dataset = clean_dataset(dataset)
    execution_time = (time.time() - start_time)
    print('Cleaned dataset in seconds: ' + str(execution_time))

    # create stopwords list
    stop_words = set(stopwords.words('english')).union(set(NON_WORD_TOKENS.values()))

    # initialize vocabulary with cleaned data, a frequency threshold and a set of stopwords
    v = Vocabulary(dataset=cleaned_dataset,
                   frequency_threshold=FREQUENCY_THRESHOLD,
                   stop_words=stop_words)
    # create vocabulary
    v.create_vocabulary()
    # save vocabulary to file
    write_to_json(object=v.vocab_output_json,
                  name=f'vocab_output_threshold_{v.threshold}_.json',
                  path=OUTPUT_VOCAB_PATH)

    # save lexicon to files
    write_to_file(text=('\n').join(list(v.vocab.keys())),
                  name=f'nyt_covid_threshold_{v.threshold}.lexicon.txt',
                  path=OUTPUT_LEXICONS_PATH)

    execution_time = (time.time() - start_time)
    print('Created vocab in seconds: ' + str(execution_time))
    """
    ###############
    NOTE:
    WE OMIT THE FOLLOWING SECTION, 
    WHICH USED THE VOCABULARY TO CREATE CORPORA FOR SIRLM
    
    # create corpora
    v.create_corpora()
    # save corpora to files
    corpus_prefix = f'nyt_covid_threshold_{v.threshold}'
    write_to_file(text=v.corpus_train,
                  name=f'{corpus_prefix}.train.txt',
                  path=OUTPUT_CORPUS_PATH)
    write_to_file(text=v.corpus_valid,
                  name=f'{corpus_prefix}.valid.txt',
                  path=OUTPUT_CORPUS_PATH)
    write_to_file(text=v.corpus_test,
                  name=f'{corpus_prefix}.test.txt',
                  path=OUTPUT_CORPUS_PATH)

    execution_time = (time.time() - start_time)
    print('Created corpus in seconds: ' + str(execution_time))
    
    #############
    """
    # print(v.stop_words)
    # print(f"vocab before threshold : {v.vocab_initial}")
    # print(f"frequency threshold : {v.threshold}")
    # print(f"vocab size : {v.vocab_size}")
    # print(f"vocab after threshold : {v.vocab}")
    # print(list(v.vocab.keys()))
    # print(f"train set : {v.train}")
    # print(f"valid set : {v.valid}")
    # print(f"test set : {v.test}")

    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))