import json
import numpy as np
from nltk.util import ngrams
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, TweetTokenizer

tokenizer = TweetTokenizer()
stop_words = set(stopwords.words('english'))
punctuation = [p for p in punctuation]

def load_jsonl(path):
    with open(path, 'r') as f:
        data = []
        for line in f:
            d = json.loads(line)
            data.append(d)
    return data

def save_jsonl(data, path):
    with open(path, 'w+') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')

def preprocess_sent(s):
    s = s.replace('\n', ' ')
    s = s.replace('\\', '')
    s = ' '.join(s.split())
    return s

def split_sent(sent, min_len = 1, remove_stopwords = False, remove_puncts = False):
    # min len is minimum length of each word in characters
    words = [w for w in tokenizer.tokenize(sent) if len(w) >= min_len]
    if remove_stopwords:
        words = [w for w in words if w not in stop_words]
    if remove_puncts:
        words = [w for w in words if w not in punctuation]
    return words

def get_ngrams(sent, n, remove_stopwords = True, remove_puncts = True):
    words = split_sent(sent, remove_stopwords = remove_stopwords, remove_puncts = remove_puncts)
    return [g for g in ngrams(words, n)]

def ngram_overlap(s1, s2, n):
    if len(s1) == 0 or len(s2) == 0:
        return 0
    g1 = s1 if isinstance(s1[0], tuple) else get_ngrams(s1, n)
    g2 = s2 if isinstance(s2[0], tuple) else get_ngrams(s2, n)
    g1 = set(g1)
    g2 = set(g2)
    overlaps = g1.intersection(g2)
    return len(overlaps)

def split_doc(doc, min_len = 3, metadata_check = 10, metadata = ["Transcriber's Note", 'U.S. copyright on this publication']):
    # min_len: in number of words
    sents = sent_tokenize(doc)
    sents = [preprocess_sent(s) for s in sents]
    sents = [s for s in sents if len(split_sent(s, remove_stopwords = True, remove_puncts = True)) >= min_len]
    # remove metadata
    sentences_with_metadata = []
    n = min(metadata_check, len(sents))
    for i in range(n):
        for md in metadata:
            if md in sents[i]:
                sentences_with_metadata.append(i)
                break
    sents = [sents[i] for i in range(n) if i not in sentences_with_metadata] + sents[metadata_check:]
    return sents

def syntactic_fidelity(query, sents, beta = 1, N = 3):
    score = 0
    for n in range(1, N + 1):
        query_ngrams = get_ngrams(query, n)
        # if len(query_ngrams) == 0:
        #     print(n, query)
        for s in sents:
            if isinstance(s[0], list):
                score += ((beta**n)*ngram_overlap(query_ngrams, s[n-1], n))
            else:
                score += ((beta**n)*ngram_overlap(query_ngrams, s, n))
    return np.sqrt(score) 
