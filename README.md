# Submodular Subset Selection for Long-Form Question Answering

This repository houses the experiments [we](#team-members) conducted to evaluate the effectiveness of Submodular Subset Selection for Long-Form Question Answering, as part of the CS 769 course project at IIT Bombay (2023 offering) under the guidance of [Prof. Ganesh Ramakrishnan](https://www.cse.iitb.ac.in/~ganesh/).

## Team Members

- Adithya Bhaskar (190050005)
- Harshit Varma (190100055)

## Directory Structure

The requirements under `requirements.txt` may be installed with 
```
pip install -r requirements.txt
```
with the exception of the `submodlib` library for the installation of which we defer instructions to its [repository](https://github.com/decile-team/submodlib). Configuration parameters for subset selection can be modified in `config.py`, while global variables are in `utils/globals.py`. Submodular optimization functions defined in `utils/submodular.py` call upon embedders in `models/embedder.py` to embed sentences. In turn, helper functions in `models/subset_selection.py` use the assistance of `utils/data.py` to load the data, then call the appropriate functions in `utils/submodular.py` to select and save spans of sentences. The file `main.py` brings together all the pieces of subset selection and allows one to change the objective used for the same. Once the subsets have been selected for each data instance, scripts in `training/scripts/` may be used to prepare the same for model training or inference. Both proceed by calling `training/lrqa/run_lrqa.py` with a configuration file inside `training/configs/`. Prior to calling `run_lrqa.py`, [add](https://stackoverflow.com/a/39022669) `training/` to your `PYTHONPATH`, if not calling from the same directory.

> **_Note:_**  As GitHub has restrictions on repository size, we are unable to upload the datasets or the subsets we obtain from our methods. Running `main.py` with the appropriate parameters generates the subsets under `data/`, and the scripts in `training/scripts/` (once modified to use the above paths) prepare the training data under `training/data`.

## Task Description

Long-Form Question Answering involves answering questions based on documents that are too long to fit into the context of standard Transformer-based models. Conventionally, special architectures have been developed to tackle this task that allow long inputs. However, an alternative approach that is also popular is to select a subset of sentences or paragraphs (called *spans* henceforth) from the source document, and pass that to a much more readily fine-tuned model (in our case, DeBERTaV3 [[1]](#1)). The filtration of spans is typically done by ranking them based on similarity to the query in terms of a metric such as TF-IDF or BM25, and then selecting the top $k$, where $k$ is a pre-determined parameter.

## Dataset

We use the QuALITY dataset [[2]](#2) in our experiments. The `train` and `dev` splits of QuALITY are publicly available, and have the following characteristics:
<div align="center">

| Split | No. of Articles | Avg. Article Length (Characters) | No. of Questions |
| :---: | :---: | :---: | :---: |
| train | 300 | 24348.86 | 2523 |
| dev | 230 | 24306.59 | 2086 |
</div>

From the dev split, we select `625` questions randomly to form the validation split, and `1461` to form the test split. Henceforth, we shall refer to the former as the `dev` split and the latter as the `test` split, at the risk of a mild abuse of definition.

Each question of the QuALITY dataset is of the multiple-choice type, with exactly four options. As QuALITY consists largely of text from novels, most questions require knowledge that can only be pieced together from numerous spans, and are rather hard even for time-pressed human annotators.

## Background

Given a ground set $V$, a submodular function $f$ maps each set $A \subseteq 2^{|V|}$ of $V$ to a real number $x$, such that for any two sets $A \subseteq B \subseteq V$ and element $x \in V$, we have

$$f(A \cup \{x\}) - f(A) \geq f(B \cup \{x\}) - f(B)$$

It turns out that submodular functions possess helpful properties that allow efficient algorithms for their optimization in the face of budgetary constraints. Readers are referred to sources such as the CS 769 course notes of IIT Bombay or this [tutorial](https://theory.stanford.edu/~jvondrak/data/submod-tutorial-1.pdf) for more information. It will suffice here to say that Greedy methods are provably near-optimal for maximizing monotone submodular functions.

Of special interest to us is that a number of submodular (and often, monotone) objectives may be defined on any arbitrary similarity function that capture either the faithfulness of a selection to a query or the extent to which it represents the ground set of spans. We are, in particular, interested here in the objectives that are based on Submodular Mutual Information.

## Approach and Optimization Objectives

We use the submodlib [[4]](#4) library for the implementations of the submodular objectives. More specifically, we evaluate the Mutual Information based versions of the [Facility Location](https://submodlib.readthedocs.io/en/latest/functions/facilityLocationMutualInformation.html), [Graph Cut](https://submodlib.readthedocs.io/en/latest/functions/graphCutMutualInformation.html) and [Log Determinant](https://submodlib.readthedocs.io/en/latest/functions/logDeterminantConditionalGain.html) objectives, along with the [Concave-over-Modular](https://submodlib.readthedocs.io/en/latest/functions/concaveOverModular.html) objective. Note that for the case of a single query (which holds always for us), Graph Cut is equivalent to choosing the greedy selection of the most similar spans. For the latter two, we also experiment with an initial filtration step which generates a selection $2.5\times$ the budget greedily before passing it to the submodular optimization. 

We use Facebook DPR [[5]](#5) to generate embeddings for spans prior to optimization, although we also support Sentence Transformers [[6]](#6). Our spans are contiguous sets of up to five sentences. In addition, we include TF-IDF and BM25 based greedy baselines for comparison. We also support the use of other objectives with TF-IDF and BM25 based embeddings, but we do not compare to them, as we found them to anecdotally perform similar to the greedy objective. We use [sklearn](https://scikit-learn.org/stable/about.html) for the former, and the rank-bm25 [[7]](#7) Python library for the latter. Our budget is $10$ spans in all cases. We also compare to empty and randomly chosen contexts with the same budget.

We noticed that the best-performing methods at the QuALITY leaderboard [[3]](#3) included a pre-training step on the RACE [[8]](#8) dataset, so we also include this step for all of our approaches. We also include a fine-tuning step on the `train` split of QuALITY for each approach on selections, based on the same approach. Our training code is a modified version of the LRQA [[9]](#9) code. The hyperparameters for each can be found under the corresponding configuration file inside `training/configs/`.

## Results

The accuracies we obtained on our `test` split for various approaches are as follows:

<div align="center">

| Approach | Accuracy (%) |
| :---: | :---: | 
| Graph Cut | 55.10 | 
| Facility Location | 50.24 | 
| Log Determinant | 54.55 |
| Filtration + Log Determinant | 52.29 |
| Concave-Over-Modular | 55.99 |
| Filtration + Concave-Over-Modular | **56.14** |
| BM25 | 52.91 |
| TF-IDF | 55.44 |
| Random | 50.38 |
| No context | 44.55 |

</div>

We would like to make a few remarks on the above observations. First, we notice that even without any context, DeBERTAV3 achieves pretty strong performance. Second, adding a randomly chosen context of 10 spans (around 40-50 sentences) helps as much as going from a random baseline to the best performing system. However, our numbers for the Graph Cut (greedy most-similar) objective roughly match those reported on the leaderboard under the "DeBERTaV3 + DPR" heading. We observe that Concave-Over-Modular, especially with filtration, outperforms Graph Cut and gets the best results, albeit by a small margin. Most submodular approaches outperform BM25, but TF-IDF surprisingly achieves similar scores. We also observe that Facility Location performs particularly poorly, as it highly prefers diversity over query-relevance.

## References
<a id="1">[1]</a> [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543), Pengcheng He, Jianfeng Gao, Weizhu Chen

<a id="2">[2]</a> [QuALITY: Question Answering with Long Input Texts, Yes!](https://arxiv.org/abs/2112.08608), Pang et. al.

<a id="3">[3]</a> QuALITY [Leaderboard](https://nyu-mll.github.io/quality/), last accessed April 26, 2023.

<a id="4">[4]</a> [Submodlib: A Submodular Optimization Library](https://arxiv.org/abs/2202.10680), Kaushal, Ramakrishnan and Iyer

<a id="5">[5]</a> [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906), Karpukhin et. al.

<a id="6">[6]</a> [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084), Reimers and Gurevych

<a id="7">[7]</a> The [rank-bm25](https://pypi.org/project/rank-bm25/) Python library, last accessed April 26, 2023.

<a id="8">[8]</a> [RACE: Large-scale ReAding Comprehension Dataset From Examinations](https://arxiv.org/abs/1704.04683), Lai et. al.

<a id="9">[9]</a> The [LRQA](https://github.com/zphang/lrqa) repository, last accessed April 26, 2023.