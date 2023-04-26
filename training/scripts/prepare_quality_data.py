import os
import json
import sys
import random

import nltk 
from nltk.tokenize import sent_tokenize

from unidecode import unidecode

in_dir_ = "/raid/infolab/adithyabhaskar/submodopt/submission/" + \
    "data/subsets/quality/dpr/conModF/"
word_limit = 320

def form_context_based_on_char_limit(groups, already_in_sentences=False, \
    char_limit=2000):
    context = ""
    remainder = char_limit
    for group in groups:
        group = unidecode(group).replace("\n", " ")
        group = " ".join(group.split())
        if len(group) <= remainder:
            remainder -= len(group)
            if context != "":
                context += " "
            context += group
        else:
            if not already_in_sentences:
                for sentence in sent_tokenize(group):
                    sentence = sentence.strip()
                    if len(sentence) <= remainder:
                        remainder -= len(sentence)
                        if context != "":
                            context += " "
                        context += sentence 
                    else:
                        break
            break
    return context
    
def save_quality(in_dir=None, suffix=""):  
    global in_dir_
    if in_dir is None:
        in_dir = in_dir_ 
    indices = {
        split: json.load(open('data/quality/{}-indices.json'.format(split))) \
        for split in ['dev', 'test']
    }
    
    outputs = {split: [] for split in ['train', 'dev', 'test']}
            
    for split in ['train', 'dev']:
        outputs_split = []
        data = json.load(open(os.path.join(in_dir, "{}{}.json".format(split, \
            suffix))))
        keys = data[0]['questions'][0].keys()
        key = [k for k in keys if k.startswith('selection')]
        assert(len(key) > 0)
        key = key[0]

        for i, example in enumerate(data):
            for j, question in enumerate(example['questions']):
                entry = {}
                entry['context'] = form_context_based_on_char_limit(question[key])
                entry['query'] = " " + question['question']
                for k, option in enumerate(question['options']):
                    entry['option_{}'.format(k)] = " " + option
                if 'gold_label' in question:
                    entry['label'] = question['gold_label']-1
                if split == 'train':
                    outputs['train'].append(entry)
                elif [i,j] in indices['dev']:
                    outputs['dev'].append(entry)
                else:
                    outputs['test'].append(entry)
    for split in ['train', 'dev', 'test']:
        print("{} examples in the {} split.".format(len(outputs[split]), split))
        out_path = "data/quality/{}.jsonl".format(split)
        with open(out_path, 'w+') as outfile:
            for entry in outputs[split]:
                json.dump(entry, outfile)
                outfile.write('\n')

    config = {
        "num_choices": 4
    }        

    with open('data/quality/config.json', 'w+') as f:
        json.dump(config, f)
        
save_quality(suffix="-n10")