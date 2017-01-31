#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield and Laure Thompson
import numpy as np
import sys
from scipy.sparse import csr_matrix, linalg
from scipy import stats
from sklearn.feature_extraction import text

# Assume corpus_mat is a csr_matrix
def run_lsa(corpus_mat, k):
    n_docs, n_words = corpus_mat.shape
    u, s, vt = linalg.svds(corpus_mat, k)

    loss_mat = corpus_mat - csr_matrix(u.dot(np.diag(s)).dot(vt))
    norm_loss = linalg.norm(loss_mat, axis=1)
    return u, s, vt.T, norm_loss


def find_W(U, s):
    return U.dot(s)


def compute_entropies(W):
    n_docs, n_dims = W.shape
    entropies = np.zeros(n_docs)
    for i in range(n_docs):
        entropies[i] = stats.entropy(np.absolute(W[i,:]))
    return entropies


if __name__ == "__main__":
    input_fname = sys.argv[1]
    k = int(sys.argv[2])
    out_pfx = sys.argv[3]

    corpus = []
    tags = []
    labels = []
    with open(input_fname) as f:
        for line in f:
            tag, label, doc = line.split('\t')
            corpus.append(doc.strip())
            tags.append(tag)
            labels.append(bool(label))
    tfidfer = text.TfidfVectorizer()
    corpus_mat = tfidfer.fit_transform(corpus)
    del corpus

    U, s, V, norm_loss = run_lsa(corpus_mat, k)
    W = find_W(U, s)

    vocab_writer = open('{}.lsa-vocab.txt'.format(out_pfx), mode='w', encoding='utf8')
    loss_writer = open('{}.lsa-loss.txt'.format(out_pfx), mode='w', encoding='utf8')
    w_writer = open('{}.lsa-w.txt'.format(out_pfx), mode='w', encoding='utf8')

    # write vocab index file
    for i, term in enumerate(tfidfer.get_feature_names()):
        vocab_writer.write('{} {}\n'.format(i, term))
    vocab_writer.close()

    # write loss and W file
    for i, tag in enumerate(tags):
        loss_writer.write('{} {} {} {}\n'.format(i, tag, labels[i], norm_loss[i]))
        w_writer.write('{} {} {} {}\n'.format(i, tag, labels[i], ' '.join(map(str, W[i,:].getA1()))))
    loss_writer.close()
    w_writer.close()

    # write usv file
    np.save('{}.usv.npy'.format(out_pfx), [U, s, V])
