from config import *
from utils.globals import *

import os
import json
import nltk
from nltk.tokenize import word_tokenize

import torch
from torch import nn
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, \
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from torch.optim import AdamW

from sentence_transformers import SentenceTransformer, util

embedding_st_model = None
dpr_context_model = None
dpr_context_tokenizer = None
dpr_question_model = None
dpr_question_tokenizer = None

def get_embedding_dpr(sentence, device=None, sentence_type="context", \
    return_tensors="pt", normalize=True, **kwargs):
    global dpr_context_model, dpr_context_tokenizer, dpr_question_model, \
        dpr_question_tokenizer
    if device is None:
        device = get_device()
    if dpr_context_model is None:
        dpr_context_model = DPRContextEncoder.from_pretrained( \
            "facebook/dpr-ctx_encoder-single-nq-base").to(device)
        dpr_context_model.eval()
        dpr_context_tokenizer = DPRContextEncoderTokenizer.from_pretrained( \
            "facebook/dpr-ctx_encoder-single-nq-base")
    if dpr_question_model is None:
        dpr_question_model = DPRQuestionEncoder.from_pretrained( \
            "facebook/dpr-question_encoder-single-nq-base").to(device)
        dpr_question_model.eval()
        dpr_question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained( \
            "facebook/dpr-question_encoder-single-nq-base")
    if sentence_type == "question":
        sent = dpr_question_tokenizer(sentence, return_tensors="pt", \
            truncation=True, max_length=512)["input_ids"].to(device)
        with torch.no_grad():
            emb = dpr_question_model(sent).pooler_output[0]
    else:
        sent = dpr_context_tokenizer(sentence, return_tensors="pt", \
            truncation=True, max_length=512)["input_ids"].to(device)
        with torch.no_grad():
            emb = dpr_context_model(sent).pooler_output[0]
    if normalize:
        emb = emb / (torch.norm(emb) + 1e-11)
    if return_tensors == "np":
        emb = emb.cpu().detach().numpy()
    return emb

def get_embedding_st(sentence, device=None, return_tensors="pt", **kwargs):
    global embedding_st_model
    if device is None:
        device = get_device()
    if embedding_st_model is None:
        embedding_st_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1').to(device)
    emb = embedding_st_model.encode([sentence])[0] # is already np
    if return_tensors == "pt":
        emb = torch.from_numpy(emb)
    return emb

def get_similarity_st(sentence1, sentence2, device=None):
    global embedding_st_model
    if device is None:
        device = get_device()
    if embedding_st_model is None:
        embedding_st_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1').to(device)
    embeddings = embedding_st_model.encode([sentence1, sentence2])
    return util.cos_sim(embeddings[0], embeddings[1]).item()

def test_embeddings(similarity_fn=get_similarity_st):
    sentence1 = "They did manage to keep a little of the past when they kept all these old things."
    sentence2 = "\"It would be nice if things were the way they used to be when I trusted Arthur--\" \"Don't you trust him now?\" Judy asked."
    sentence3 = "Come on up and I'll show you.\" Lois and Lorraine had finished their dessert while Judy was telling them the story of the fountain."
    sentence4 = "Maybe I'll find the answers to some of them when I finish sorting Grandma's things."
    sentence5 = "The STS 2014 dataset BIBREF37 consists of 3,750 pairs and ratings from six linguistic domains."
    print(similarity_fn(sentence1, sentence2))
    print(similarity_fn(sentence1, sentence3))
    print(similarity_fn(sentence1, sentence4))
    print(similarity_fn(sentence1, sentence5))

if __name__ == '__main__':
    pass