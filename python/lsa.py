#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
import numpy as np
from scipy import linalg
from scipy import stats
from sklearn.feature_extraction import text


def run_lsa(fname, k):
    corpus = []
    labels = []
    with open(fname) as f:
        for line in f:
            tag, label, doc = line.split('\t')
            corpus.append(doc.strip())
            labels.append(bool(label))
    tfidfer = text.TfidfVectorizer()
    corpus_mat = tfidfer.fit_transform(corpus)
    n_docs, n_words = corpus_mat.shape
    U, s, Vh = linalg.svd(corpus_mat)

    loss_mat = corpus_mat - U[:k,:].dot(diag(s)).dot(Vh[:k,:])
    norm_loss = np.linalg.norm(loss_mat, axis=1)
    return U[:,:k], s, Vh[:k,:].T, norm_loss


def find_W(U, s):
    return U.dot(s)


def compute_entropies(W):
    n_docs, n_dims = W.shape
    entropies = np.zeros(n_docs)
    for i in range(n_docs):
        entropies[i] = stats.entropy(np.absolute(W[i,:]))
    return entropies
    
    

