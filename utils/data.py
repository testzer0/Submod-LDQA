from config import *
from utils.globals import *

import os
import nltk
import json
from csv import reader
from nltk.tokenize import sent_tokenize, word_tokenize
from unidecode import unidecode

QUALITY_ROOT = os.path.join(DATASETS_ROOT, "QuALITY", "htmlstripped")

def load_quality(split='train'):
    if split == 'validation':
        split = 'dev'
    file_path = os.path.join(QUALITY_ROOT, "{}.json".format(split))
    return json.load(open(file_path))

def get_quality_raw_texts(split='train', min_sent_len=1):
    quality = load_quality(split)
    texts = []
    
    # Quality uses '\n' for line breaks as in the book even where the 
    # sentences don't end.
    for example in quality:
        text = example['article'].replace('\n', ' ')
        text = " ".join(text.split())
        sentences = sent_tokenize(text)
        # Throw away sentences that are too short during training.
        sentences = [sent.strip() for sent in sentences if len(word_tokenize(sent)) >= \
            min_sent_len]
        texts.append([unidecode(sentence) for sentence in sentences])
        
    return texts

if __name__ == '__main__':
    pass