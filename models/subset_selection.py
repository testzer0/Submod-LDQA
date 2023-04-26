from config import *
from utils.globals import *

from utils.data import get_quality_raw_texts, load_quality
from utils.submodular import *
from models.embedder import get_embedding, get_embedding_st, get_embedding_dpr

import os
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from unidecode import unidecode

def get_sentences(text, min_sent_len=0):
    text = text.replace('\n', ' ')
    text = " ".join(text.split())
    sentences = sent_tokenize(text)
    sentences = [sent.strip() for sent in sentences if len(word_tokenize(sent)) >= \
        min_sent_len]
    return [unidecode(sentence) for sentence in sentences]

def select_subset_for_question(all_sentences, question, n=30, \
    embedder_fn=get_embedding_st, diversity=0.5, min_length=0, \
    selection_fn=select_sentences_MI, **kwargs):
    question = unidecode(question)
    return selection_fn(all_sentences, question, embedder_fn, \
        budget=n, diversity=diversity, min_length=min_length)

def select_subset_groups_for_question(all_sentences, question, n=30, \
    embedder_fn=get_embedding_dpr, diversity=1, min_length=0, \
    selection_fn=select_sentences_closest, already_grouped=False, \
    mode="graph_cut", filter_based_on_similarity=False, **kwargs):
    if not already_grouped:
        # Group sentences in sets of five
        all_sentences = [" ".join(all_sentences[i:i+5]) for i in \
            range(0,len(all_sentences),5)]
    question = unidecode(question)
    return selection_fn(all_sentences, question, embedder_fn, \
        budget=n, diversity=diversity, min_length=min_length, mode=mode, \
        filter_based_on_similarity=filter_based_on_similarity)

def save_selected_sentences_quality(out_dir, split=['train', 'dev', 'test'], \
    diversity=0.5, n=35, verbose=True, save_every=50, min_length=0, \
    selection_fn=select_sentences_MI, embedder_fn=get_embedding_dpr, \
    call_fn=None, mode="graph_cut", filter_based_on_similarity=False):
    if call_fn is None:
        call_fn = select_subset_for_question
    if type(split) == list:
        for s in split:
            save_selected_sentences_quality(out_dir, s, diversity, n, verbose, \
                min_length,selection_fn)
        return
    if type(diversity) == list:
        for d in diversity:
            save_selected_sentences_quality(out_dir, split, d, n, verbose, \
                min_length, selection_fn)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, split+"-n{}.json".format(n, \
        int(100*diversity)))
    quality = load_quality(split)
    if verbose:
        print("Starting QuALITY [{}] subset selection...".format(split))
        print("Diversity = {:.2f}".format(diversity))
        print("Top-{} sentences for each question\n".format(n))
    for i in tqdm(range(len(quality))):
        all_sentences = get_sentences(quality[i]['article'])
        questions = []
        for question in quality[i]['questions']:
            question['selections_{:.2f}'.format(diversity)] = \
                call_fn(all_sentences, question['question'], \
                    n=n, embedder_fn=embedder_fn, diversity=diversity, \
                    min_length=min_length, selection_fn=selection_fn, mode=mode, \
                    filter_based_on_similarity=filter_based_on_similarity)
            questions.append(question)
        quality[i]['questions'] = questions
        if (i+1) % save_every == 0:
            json.dump(quality, open(out_path, 'w+'))
    if verbose:
        print("------------------------")
    json.dump(quality, open(out_path, 'w+'))

def maybe_split_paragraph(paragraph, max_words=100, max_chars=500, hard_limit=120):
    paragraph = unidecode(paragraph).replace("\n", " ")
    paragraph = " ".join(paragraph.split())
    n_words = len(paragraph.split(" "))
    if n_words <= max_words and len(paragraph) <= max_chars:
        return [paragraph]
    sentences = sent_tokenize(paragraph)
    n_parts = (n_words+max_words-1) // max_words
    part_size_limit = max((n_words+n_parts-1) // n_parts, 75)
    parts = []
    current = ""
    leeway = part_size_limit
    leeway_chars = max_chars
    for sentence in sentences:
        sent_words = len(sentence.split(" "))
        if leeway > sent_words and leeway_chars > len(sentence):
            if current != "":
                sentence = " " + sentence
            current += sentence
            leeway -= sent_words
            leeway_chars -= len(sentence)
        elif sent_words <= part_size_limit and len(sentence) <= max_chars:
            if current != "":
                parts.append(current)
            current = sentence
            leeway = part_size_limit - sent_words
            leeway_chars = max_chars - len(sentence)
        else:
            if current != "":
                parts.append(current)
            if sent_words < hard_limit and len(sentence) < max_chars:
                parts.append(sentence)
            current = ""
            leeway = part_size_limit
            leeway_chars = max_chars
    if current != "":
        parts.append(current)
    return parts

def smooth_paragraphs(paragraphs):
    smoothed = []
    for paragraph in paragraphs:
        smoothed += maybe_split_paragraph(paragraph)
    return smoothed

def get_sentences(paragraphs):
    sentences = []
    for paragraph in paragraphs:
        sentences += sent_tokenize(unidecode(paragraph))
    return sentences

if __name__ == '__main__':
    pass