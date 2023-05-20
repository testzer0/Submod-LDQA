import sys
import time
import utils
import random
import numpy as np
from tqdm import tqdm
from functools import partial

SELECTOR = 'synfid'
IN_DIR = 'data'
OUT_DIR = f'selected_data_{SELECTOR}'

def select_random(sents, K):
    selected_indices = sorted(random.sample(list(range(len(sents))), K))
    selected_sentences = [sents[i] for i in selected_indices]
    return selected_sentences

def greedy(query, sents, sents_ngrams, f, K):
    selected_indices = []
    selected_sentences = []
    for k in range(K):
        k_new = 0
        M = -1
        for i, s in enumerate(sents):
            if i not in selected_indices:
                v = f(query, selected_sentences + [sents_ngrams[i]])
                if v > M:
                    k_new = i
                    M = v
        selected_indices.append(k_new)
        selected_sentences.append(sents[k_new])
    selected_sentences = [sents[i] for i in sorted(selected_indices)]
    return selected_sentences

if __name__ == '__main__':

    split = sys.argv[1]
    data = utils.load_jsonl(f'{IN_DIR}/{split}.jsonl')

    # submodular parameters
    num_select = 10
    max_ngrams = 3
    beta = 1.5
    f = partial(utils.syntactic_fidelity, beta = beta, N = max_ngrams)

    doc2sents = {}

    for i, d in enumerate(tqdm(data)):

        article_id = d['article_id']
        question_id = d['question_id']
        query = utils.preprocess_sent(d['query'])
        context = d['context']

        if SELECTOR == 'synfid':

            try:
                sents, sents_ngrams = doc2sents[article_id]
                # print('reusing')
            except:
                sents = utils.split_doc(context)
                sents_ngrams = [
                    [utils.get_ngrams(s, n) for n in range(1, max_ngrams + 1)]
                    for s in sents
                ]
                doc2sents[article_id] = (sents, sents_ngrams)

            if len(sents) <= num_select:
                selected = sents
            else:
                selected = greedy(query, sents, sents_ngrams, f, num_select)
                assert len(selected) == num_select

        elif SELECTOR == 'random':

            try:
                sents = doc2sents[article_id]
                # print('reusing')
            except:
                sents = utils.split_doc(context)
                doc2sents[article_id] = sents
            selected = select_random(sents, num_select)

        data[i]['context'] = ' '.join(selected)

    utils.save_jsonl(data, f'{OUT_DIR}/{split}.jsonl')