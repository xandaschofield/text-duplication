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

    print('{}: ...calculating U*s'.format(k), file=sys.stderr, flush=True)
    us = u.dot(np.diag(s))
    
    print('{}: ...calculating loss'.format(k), file=sys.stderr, flush=True)
    norm_loss = np.zeros(n_docs)
    for i in range(n_docs):
        if not i % 1000:
            print('{}: ......{}/{} row'.format(k, i, n_docs), file=sys.stderr, flush=True)
        approx_row = us[i, :].dot(vt)
        loss = corpus_mat.getrow(i).todense() - approx_row
        norm_loss[i] = np.linalg.norm(loss)

    #norm_loss = linalg.norm(loss_mat, axis=1)
    return u, s, vt.T, norm_loss


def find_W(U, s):
    return U.dot(np.diag(s))

def read_in_w(w_fname):
    repeats_mask = []
    w_rows_list = []
    with open(w_fname) as w_file:
        for i, line in enumerate(w_file):
            row = line.strip().split()
            _, original_id, line_id = row[1].split('-')
            label = (row[2] == 'True')
            repeats_mask.append(label)
            w_row = np.array([float(c) for c in row[3:]])
            w_rows_list.append(w_row)
    repeats_mask = np.array(repeats_mask, dtype=bool)
    w_matrix = np.vstack(w_rows_list)
    return repeats_mask, w_matrix

def read_in_loss(loss_fname):
    repeats_mask = []
    loss = []
    with open(loss_fname) as loss_file:
        for i, line in enumerate(loss_file):
            row = line.strip().split()
            label = (row[2] == 'True')
            repeats_mask.append(label)
            loss.append(float(row[3]))
    repeats_mask = np.array(repeats_mask, dtype=bool)
    loss = np.array(loss, dtype=float)
    return repeats_mask, loss

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
    print("{}: Reading input file...".format(k), file=sys.stderr, flush=True)
    with open(input_fname) as f:
        for line in f:
            tag, label, doc = line.split('\t')
            corpus.append(doc.strip())
            tags.append(tag)
            labels.append(label == 'True')
    print("{}: Computing tfidf matrix...".format(k), file=sys.stderr, flush=True)
    tfidfer = text.TfidfVectorizer()
    corpus_mat = tfidfer.fit_transform(corpus)
    del corpus # attempting to reduce memory

    print("{}: Running LSA method...".format(k), file=sys.stderr, flush=True)
    U, s, V, norm_loss = run_lsa(corpus_mat, k)
    del corpus_mat # attempting to reduce memory usage
    print("{}: Calculating W".format(k), file=sys.stderr, flush=True)
    W = find_W(U, s)

    print('{}: Starting all the file writing...'.format(k), file=sys.stderr, flush=True)
    vocab_writer = open('{}.lsa.vocab.txt'.format(out_pfx), mode='w', encoding='utf8')
    loss_writer = open('{}.lsa.loss.txt'.format(out_pfx), mode='w', encoding='utf8')
    w_writer = open('{}.lsa.w.txt'.format(out_pfx), mode='w', encoding='utf8')

    # write vocab index file
    for i, term in enumerate(tfidfer.get_feature_names()):
        vocab_writer.write('{} {}\n'.format(i, term))
    vocab_writer.close()

    # write loss and W file
    for i, tag in enumerate(tags):
        label = labels[i]
        loss_writer.write('{} {} {} {}\n'.format(i, tag, label, norm_loss[i]))
        w_string = ' '.join(map(str, W[i, :]))
        w_writer.write('{} {} {} {}\n'.format(i, tag, label, w_string))
    loss_writer.close()
    w_writer.close()

    # write usv file
    np.save('{}.usv.npy'.format(out_pfx), [U, s, V])
