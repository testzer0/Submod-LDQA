from config import *
from utils.globals import *

import os
import json
import random
import numpy as np

from submodlib.functions.facilityLocationMutualInformation import \
    FacilityLocationMutualInformationFunction
from submodlib.functions.graphCutMutualInformation import \
    GraphCutMutualInformationFunction
from submodlib.functions.logDeterminantMutualInformation import \
    LogDeterminantMutualInformationFunction
from submodlib.functions.concaveOverModular import \
    ConcaveOverModularFunction

from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

def get_sims(data: np.ndarray, nonnegative: bool = True, eps: float = 1e-10) \
    -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data /= (norms + eps)
    sims = np.matmul(data, data.T)
    # At this point, the similarities are from [-1, 1]. Convert this to [0, 1]
    if nonnegative:
        sims = (1 + sims) / 2
    return sims

def get_context_sims(data: np.ndarray, context: np.ndarray, \
    nonnegative: bool = True, eps: float = 1e-10) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data /= (norms + eps)
    if context.ndim == 1:
        context_norm = np.linalg.norm(context)
        context /= context_norm + eps
        context_sims = np.matmul(data, np.expand_dims(context, axis=1)).squeeze()
    else:
        assert(context.ndim == 2)
        context_norm = np.linalg.norm(context, axis=1)
        context /= context_norm + eps
        context_sims = np.matmul(data, context.T)
        context_sims = np.mean(context_sims, axis=-1)
    if nonnegative:
        context_sims = (1 + context_sims) / 2
    return context_sims

class ContextualFacilityLocation:
    def __init__(self, sims=None, context_sims=None, data=None, context=None, \
        budget=None, diversity=0.5, nonnegative=True):
        assert((sims is not None and context_sims is not None) or (data is not \
            None and context is not None))
        if data is not None:
            assert(data.ndim == 2 and data.shape[-1] == context.shape[-1])
            self.sims = get_sims(data, nonnegative=nonnegative)
            self.context_sims = get_context_sims(data, context, \
                nonnegative=nonnegative)
        else:
            self.sims = sims
            self.context_sims = context_sims
        self.n_ground = self.sims.shape[0]
        self.memoized_max_sims = [0 for _ in range(self.n_ground)] 
        self.selected = [False for _ in range(self.n_ground)] 
        self.current_set = []
        self.budget = budget
        self.diversity = diversity
    
    def reset(self):
        self.memoized_max_sims = [0 for _ in range(self.n_ground)] 
        self.selected = [False for _ in range(self.n_ground)] 
        self.current_set = []
    
    def evaluate(self, selection: list, budget: int) -> float:
        term1 = 0
        term2 = 0
        if budget == 0 or len(selection) > budget:
            return 0
        for idx in selection:
            term2 += self.context_sims[idx]
        term2 *= self.n_ground / budget
        
        for other_idx in range(self.n_ground):
            max_sim = 0
            for idx in selection:
                max_sim = max(max_sim, self.sims[idx, other_idx])
            term1 += max_sim
        
        return self.diversity * term1 + (1 - self.diversity) * term2
        
    def get_marginal_benefit(self, idx: int) -> float:
        assert(self.budget is not None and self.budget > 0)
        if self.selected[idx]:
            return 0
        benefit_term1 = 0
        for other_idx in range(self.n_ground):
            benefit_term1 += max(0, self.sims[idx, other_idx] - \
                self.memoized_max_sims[other_idx])
        
        benefit_term2 = self.context_sims[idx]
        benefit_term2 *= self.n_ground / self.budget
        return self.diversity * benefit_term1 + (1 - self.diversity) * \
            benefit_term2
    
    def get_element_with_max_marginal_benefit(self):
        assert(self.budget is not None and self.budget > 0)
        if len(self.current_set) >= self.budget:
            return None
        best_idx = None
        best_benefit = 0
        for idx in range(self.n_ground):
            if not self.selected[idx]:
                benefit = self.get_marginal_benefit(idx)
                if benefit  > best_benefit:
                    best_idx = idx
                    best_benefit = benefit
        return best_idx
            
    def select_element(self, idx: int):
        assert(self.budget is not None and self.budget > 0)
        if self.selected[idx]:
            return
        for other_idx in range(self.n_ground):
            self.memoized_max_sims[other_idx] = max(self.sims[idx, other_idx],
                self.memoized_max_sims[other_idx])
        self.selected[idx] = True
        self.current_set.append(idx)

    def optimize_greedy(self, budget: int = None):
        if budget is None:
            assert(self.budget is not None)
        else:
            self.budget = budget
        self.reset()
        while len(self.current_set) < self.budget:
            idx = self.get_element_with_max_marginal_benefit()
            if idx is None:
                break
            self.select_element(idx)
        return self.current_set

def select_sentences(corpus, context, embedder_fn, budget=30, diversity=0.5, \
    nonnegative=True, return_indices=False, min_length=0):
    assert(type(corpus) == list)
    if min_length > 0:
        assert(not return_indices)
        corpus = [sentence for sentence in corpus if \
            len(sentence.strip().split()) > min_length]
    data = [embedder_fn(sentence, return_tensors="np") for \
        sentence in corpus if sentence.strip() != ""]
    if len(data) == 0:
        print("Warning: empty selection out of {} sentences".format(len(corpus)))
        return []
    data = np.stack(data)
    if type(context) == list:
        context = np.stack([embedder_fn(sentence, return_tensors="np") for \
            sentence in context])
    else:
        context = embedder_fn(context, return_tensors="np")
    optimizer = ContextualFacilityLocation(data=data, context=context, \
        budget=budget, diversity=diversity, nonnegative=nonnegative)
    indices = optimizer.optimize_greedy()
    if return_indices:
        return indices
    else:
        return [corpus[ind] for ind in indices]

def get_objective(n, num_queries, data=None, queryData=None, data_sijs=None, \
    query_sijs=None, mode="facility_location"):
    cls_dict = {
        "facility_location": FacilityLocationMutualInformationFunction, 
        "log_determinant": LogDeterminantMutualInformationFunction,
        "graph_cut": GraphCutMutualInformationFunction,
        "concave_over_modular": ConcaveOverModularFunction
    }
    extra_kwargs = {
        "facility_location": {
            "magnificationEta": 1
        },
        "log_determinant": {
            "magnificationEta": 1,
            "lambdaVal": 1e-8,
            "query_query_sijs": None if query_sijs is None else np.ones((1,1))
        },
        "graph_cut": {
            
        },
        "concave_over_modular": {
            "queryDiversityEta": 1
        }
    }
    if mode == "concave_over_modular":
        return cls_dict[mode](n=n, num_queries=num_queries, data=data, \
            queryData=queryData, query_sijs=query_sijs, **extra_kwargs[mode])
    else:
        return cls_dict[mode](n=n, num_queries=num_queries, data=data, \
            queryData=queryData, data_sijs=data_sijs, query_sijs=query_sijs, \
            **extra_kwargs[mode])

def select_subset_of_indices_and_adjust_similarities(sims, budget, factor=2.5):
    limit = int(factor*budget)
    chosen = np.argsort(sims)[-limit:].tolist()[::-1]
    sims_chosen = np.array([sims[i] for i in chosen])
    sims_chosen = (sims_chosen - sims_chosen[-1]) / (sims_chosen[0] - sims_chosen[-1])
    return chosen, np.expand_dims(sims_chosen, axis=1)

def select_sentences_MI(corpus, context, embedder_fn, budget=30, \
    return_indices=False, min_length=0, mode="facility_location", diversity=1.0, \
    filter_based_on_similarity=False):
    # NOTE: here diversity is misleading, its actually the opposite of diversity
    assert(type(corpus) == list)
    if min_length > 0:
        assert(not return_indices)
        corpus = [sentence for sentence in corpus if \
            len(sentence.strip().split()) > min_length]
        corpus = [sentence for sentence in corpus if sentence.strip() != ""]
    data = [embedder_fn(sentence, sentence_type="context", \
        return_tensors="np") for sentence in corpus]
    if len(data) == 0:
        print("Warning: empty selection out of {} sentences".format(len(corpus)))
        return []
    data = np.stack(data)
    budget = min(budget, data.shape[0]-1)
    context = np.expand_dims(embedder_fn(context, sentence_type="question", \
        return_tensors="np"), axis=0)
    if filter_based_on_similarity:
        sims = np.matmul(data, context.T)[:,0]
        indices, query_sijs = select_subset_of_indices_and_adjust_similarities( \
            sims, budget, factor=2.5)
        corpus = [corpus[i] for i in indices]
        data = np.stack([data[i] for i in indices])
        budget = min(budget, data.shape[0]-1)
        data_sijs = np.matmul(data, data.T)
        objective = get_objective(n=data.shape[0], num_queries=context.shape[0], \
            data_sijs=data_sijs, query_sijs=query_sijs, mode=mode)
    else:
        objective = get_objective(n=data.shape[0], \
            num_queries=context.shape[0], data=data, queryData=context, mode=mode)
    chosen = objective.maximize(budget=budget, show_progress=False)
    indices = [idx for idx, _ in chosen]
    if return_indices:
        return indices
    else:
        return [corpus[ind] for ind in indices]

def select_sentences_closest(corpus, context, embedder_fn, budget=30, \
    return_indices=False, min_length=0, **kwargs):
    assert(type(corpus) == list)
    assert(not return_indices)
    if min_length > 0:
        corpus = [sentence for sentence in corpus if \
            len(sentence.strip().split()) > min_length]
    data = [embedder_fn(sentence, return_tensors="np", sentence_type="context") for \
        sentence in corpus]
    if len(data) == 0:
        print("Warning: empty selection out of {} sentences".format(len(corpus)))
        return []
    data = np.stack(data)
    budget = min(budget, data.shape[0])
    if type(context) == list:
        context = np.stack([embedder_fn(sentence, sentence_type="question", \
            return_tensors="np") for sentence in context])
    else:
        context = np.expand_dims(embedder_fn(context, sentence_type="question", \
            return_tensors="np"), axis=0)
    similarities = np.mean(np.matmul(data, context.T), axis=-1)
    chosen = np.argsort(similarities)[-budget:].tolist()[::-1]
    return [corpus[idx] for idx in chosen]    

def select_bm25_sentences(corpus, context, embedder_fn, budget=30, min_length=0, \
    **kwargs):
    assert(type(corpus) == list)
    if min_length > 0:
        corpus = [sentence for sentence in corpus if \
            len(sentence.strip().split()) > min_length]
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = context.split(" ")
    return bm25.get_top_n(tokenized_query, corpus, n=budget)  

def select_bm25_sentences_with_submod(corpus, context, embedder_fn, budget=30, \
    min_length=0, mode="log_determinant", filter_based_on_similarity=False, \
    **kwargs):
    assert(type(corpus) == list)
    if min_length > 0:
        corpus = [sentence for sentence in corpus if \
            len(sentence.strip().split()) > min_length]
    if len(corpus) == 0:
        return []
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    budget = min(budget, len(corpus))
    if type(context) == list:
        context = " ".join(context)
    tokenized_query = context.split(" ")
    query_sijs = np.expand_dims(bm25.get_scores(tokenized_query), axis=1)
    sijs = []
    for document in corpus:
        tokenized_document = document.split(" ")
        sijs.append(bm25.get_scores(tokenized_document))
    sijs = np.stack(sijs)
    budget = min(budget, len(corpus)-1)
    if filter_based_on_similarity:
        assert(0) # Not supported
    else:
        objective = get_objective(n=len(corpus), \
            num_queries=1, data_sijs=sijs, query_sijs=query_sijs, \
            mode=mode)
    chosen = objective.maximize(budget=budget, show_progress=False)
    indices = [idx for idx, _ in chosen]
    return [corpus[ind] for ind in indices]

def select_tf_idf_sentences(corpus, context, embedder_fn, budget=30, \
    min_length=0, **kwargs):
    assert(type(corpus) == list)
    if min_length > 0:
        corpus = [sentence for sentence in corpus if \
            len(sentence.strip().split()) > min_length]
    budget = min(budget, len(corpus))
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    all_sentences = [context] + corpus
    features = vectorizer.fit_transform(all_sentences)
    scores = (features[0, :] * features[1:, :].T).A[0]
    chosen = np.argsort(scores)[-budget:].tolist()[::-1]
    return [corpus[idx] for idx in chosen]     

def select_tf_idf_sentences_with_submod(corpus, context, embedder_fn, budget=30, \
    min_length=0, mode="log_determinant", filter_based_on_similarity=False, \
    **kwargs):
    assert(type(corpus) == list)
    if min_length > 0:
        corpus = [sentence for sentence in corpus if \
            len(sentence.strip().split()) > min_length]
    if len(corpus) == 0:
        return []
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
    all_sentences = [context] + corpus
    features = vectorizer.fit_transform(all_sentences)
    queryData = features[:1, :].A
    data = features[1:, :].A
    query_sijs = np.matmul(data, queryData.T)
    sijs = np.matmul(data, data.T)
    budget = min(budget, len(corpus)-1)
    if filter_based_on_similarity:
        assert(0) # Not supported
    else:
        objective = get_objective(n=len(corpus), \
            num_queries=1, data_sijs=sijs, query_sijs=query_sijs, \
            mode=mode)
    chosen = objective.maximize(budget=budget, show_progress=False)
    indices = [idx for idx, _ in chosen]
    return [corpus[ind] for ind in indices]

def select_random_sentences(corpus, context, embedder_fn, budget=30, \
    min_length=0, **kwargs):
    assert(type(corpus) == list)
    if min_length > 0:
        corpus = [sentence for sentence in corpus if \
            len(sentence.strip().split()) > min_length]
    budget = min(budget, len(corpus))
    return random.choices(corpus, k=budget) 
    
if __name__ == '__main__':
    pass