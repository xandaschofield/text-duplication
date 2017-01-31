#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
from collections import Counter

import numpy as np
from scipy import stats
from scipy.spatial import distance
from sklearn.feature_extraction import text


def split_docs_by_repeated(input_seq_fname)
    """Take input doc files and find which ones are vs. are not repeated,
    as well as gathering information about the unigram language model of
    the repeated documents.
    
    Input:
        input_seq_fname: the path to a text file of Mallet-formatted data
        for repeats.

    Outputs:
        repeated_documents: list of integer ids of repeated documents
        unrepeated_documents: list of integer ids of unrepeated documents
          *the union of these two should be range(25000)
        doc_models: an array of word frequencies for each repeated document
        cv.vocabulary_: a map of vocabulary word to id
    """
    repeated_docs = []
    unrepeated_docs = []
    repeated_lines = []
    with open(input_seq_fname) as f:
        for line in f:
            doc_id, is_repeat, text = line.split('\t')
            _, original_id, line_id = doc_id.split('-')
            if is_repeat == 'True' and int(original_id) not in repeated_docs:
                repeated_lines.append(line)
                repeated_docs.append(int(original_id))
            else if is_repeat == 'False':
                unrepeated_docs.append(int(original_id))
                
    cv = text.CountVectorizer()
    doc_models = cv.fit_transform(repeated_lines)
    return repeated_documents, unrepeated_documents, doc_models, cv.vocabulary_


def get_top_keys_unigram(unigram_model, vocab, n_top_keys=20):
    """Obtain the top keys from a unigram language model"""
    top_word_counter = Counter()
    for word, word_idx in vocab.items():
        top_word_counter[word] = unigram_model[word_idx]
    top_words = [k for k, v in top_word_counter.most_common(n_top_keys)]
    return top_words


def compute_entropies(doc_vecs):
    """Compute entropy from document topic vectors for LDA."""
    n_docs, n_dims = doc_vecs.shape
    entropies = np.zeros(n_docs)
    for doc_idx in n_docs:
        entropies[doc_idx] = stats.entropy(doc_vecs[doc_idx, :])
    return entropies


def compute_unigram_lm_similarities(word_weights, unigram_vec):
    """Compute the cosine similarity of each topic with a unigram
    language model."""
    n_words, n_topics = word_vecs.shape
    return np.array([
        1 - distance.cosine(word_vecs[:, k].toarray().ravel(), unigram_vec)
        for k in range(n_topics)
    ])


def compute_key_proportion(key_lists, unigram_key_list):
    """Compute what proportion of top keys are from the top keys of the unigram
    language model."""
    total_keys = sum((len(kl) for kl in key_lists))
    total_matched = sum((
        len((w for w in kl if w in unigram_key_list))
        for kl in key_lists))
    return float(total_matched)/total_keys


def compute_perplexity(log_prob_fname, mask):
    """Compute the perplexity of documents in a topic model, including
    only those matching a mask."""
    num = 0
    denom = 0
    with open(log_prob_fname) as f:
        for i, line in enumerate(f):
            if mask[i]:
                num += float(line.split()[-1])
                denom += 1
    return np.exp(-num / denom)


