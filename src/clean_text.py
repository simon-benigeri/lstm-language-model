import os
import re
import time
from typing import List, Dict
from nltk.corpus import stopwords

METADATA_TAGS = {
    '_titel_': '<<start_title>>',  # Start-of-title token
    '_/titel_': '<<end_title>>',  # End-of-title token
    '_pub_time_': '<<start_pub_time>>',  # start-of-publication-time token
    '_/pub_time_': '<<end_pub_time>>',  # start-of-publication-time token
    '_articel_id_': '<<start_article_id>>',  # start-of-article-id token
    '_/articel_id_': '<<end_article_id>>',  # end-of-article-id token
    '_despriction_': '<<start_desc>>',  # start-of-description token
    '_/despriction_': '<<end_desc>>',  # end-of-description token
    '_news_keywords_': '<<start_keywords>>',  # start-of-keywords token
    '_/news_keywords_': '<<end_keywords>>',  # end-of-keywords token
    '_sect_': '<<start_section>>',  # start-of-section token
    '_/sect_': '<<end_section>>',  # end-of-section token
    '_s_': '',# '<s>',  # start-of-sentence token
    '_/s_': '</s>',  # end-of-sentence token
    '_start_article_': '<<start_article>>',  # Start-of-article token
    '_end_article_': '<<end_article>>',  # End-of-article token
    '_start_article_body_': '<<start_body>>',  # Start-of-article-body token
    '_end_article_body_': '<<end_body>>'  # End-of-article-body token
    # '_end_article_body': '<<end_body>>'        # Badly spelt End-of-article-body token
}

def _read_text(path: str) -> str:
    with open(path, 'r') as f:
        text = f.read()  # f.readlines()
    return text


def load_data(path: str) -> Dict:
    dataset = {
        'train': _read_text(os.path.join(path, 'train.txt')),
        'valid': _read_text(os.path.join(path, 'valid.txt')),
        'test': _read_text(os.path.join(path, 'test.txt'))
    }
    return dataset


def _substitute_tags(text: str, dictionary: Dict) -> str:
    # sort keys by length, in reverse order
    for item in sorted(dictionary.keys(), key=len, reverse=True):
        text = re.sub(item, dictionary[item], text)
    return text


def _select_content_between_tags(text: str, start_tag: str, end_tag: str) -> str:
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

def write_to_file(text: str, name: str, path: str):
    with open(os.path.join(path, name), 'w') as f:
        f.write(text)
    return

if __name__ == "__main__":
    start_time = time.time()

    # set paths and hyper param
    INPUT_PATH = '../data/raw_text_files'  # '../data/nyt_small_samples' #

    OUTPUT_CORPUS_PATH = '../data/corpora'
    # load datasets
    dataset = load_data(path=INPUT_PATH)
    # clean dataset
    cleaned_dataset = clean_dataset(dataset)
    execution_time = (time.time() - start_time)
    print('Cleaned dataset in seconds: ' + str(execution_time))
    for file in cleaned_dataset:
        write_to_file(text=cleaned_dataset[file], name=f'nyt_covid.{file}.txt', path=OUTPUT_CORPUS_PATH)
    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))