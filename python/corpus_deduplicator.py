#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield

"""corpus_deduplicator.py

An class for deduplicating corpora.
"""
import sys

from nltk.tokenize import RegexpTokenizer
import numpy as np
from scipy.spatial import distance
from sklearn.feature_extraction import text


class CorpusDeduplicator(object):

    def __init__(self,
            token_pattern=r"\w[\w\-']+\w",
            max_df=0.8,
            diff_tol=0.2,
            max_ngram=12
        ):
        """Creates a new CorpusDeduplicator.

        Arguments:
            token_pattern::string -- regular expression for tokens
            max_df::float -- maximum document frequency proportion (0 to 1)
            diff_tol::float -- the minimum proportion of different vocabulary
                required to be considered a unique document
        """
        self.token_pattern = token_pattern
        self.max_df = max_df
        self.max_ngram = max_ngram
        self.diff_tol = diff_tol
        self.doc_lines = []
        self.vocab = {}

    def load_corpus_from_file(self, filename):
        """Load a MALLET corpus from a file, retaining only unigram counts.
        Assumes one line per document

        Arguments:
            filename::str -- name of the file from which to read documents

        Returns:
            a matrix of document word counts of shape (# documents, # vocab)
        """
        corpus_cv = text.CountVectorizer(
                token_pattern=self.token_pattern,
                max_df=self.max_df
        )
        with open(filename) as corpus_file:
            self.doc_lines = [line for line in corpus_file]
        self.docs_to_keep = set(range(len(self.doc_lines)))
        self.raw_document_vectors = corpus_cv.fit_transform(self.doc_lines).tocsr()
        self.vocab = corpus_cv.vocabulary_

    def deduplicate_bow(self):
        """Returns a set of document indices to retain based upon removing
        texts with high unigram overlap, retaining only the longest in each
        overlap check.

        Returns:
        a set of integers identifying indices of documents to retain
        """
        doc_lengths = self.raw_document_vectors.sum(axis=1)
        n_docs = len(doc_lengths)

        buckets = [0.1 * i for i in range(1, 10)]
        numerators = [0 for i in range(1, 10)]
        denominator = 0

        # The comparison matrix here is pretty big, so we do this in
        # chunks to compare documents with each other
        stride = 2500
        for i in range(0, n_docs, stride):
            iend = min(i + stride, n_docs)
            for j in range(i, n_docs, stride):
                print('Index', i, j)
                # We gather indices of documents with less than a tolerated
                # cosine distance between them
                jend = min(j + stride, n_docs)
                comp_mat = distance.cdist(
                        self.raw_document_vectors[i:iend,:].toarray(),
                        self.raw_document_vectors[j:jend,:].toarray(),
                        'cosine'
                )
                denominator += comp_mat.shape[0] * comp_mat.shape[1]
                for c, b in enumerate(buckets):
                    numerators[c] += len(np.where(comp_mat < b)[0])
                xs, ys = np.where(comp_mat < self.diff_tol)
                for x, y in zip(xs, ys):
                    if x == y or np.isnan(comp_mat[x, y]):
                        continue
                    # We keep the longer of the two documents
                    idx_a = x + i
                    idx_b = y + j
                    if idx_a in self.docs_to_keep and idx_b in self.docs_to_keep:
                        if doc_lengths[idx_b] > doc_lengths[idx_a]:
                            self.docs_to_keep.remove(idx_a)
                        else:
                            self.docs_to_keep.remove(idx_b)
        print('Dedupe bucket')
        for b, num in zip(buckets, numerators):
            print(b, ':', float(num - n_docs) / (denominator - n_docs))

    def deduplicate_ngrams(self):
        """Modifies the lines of text to remove long n-grams that appear
        frequently. Edits self.doc_lines in place.
        """
        trie = {}
        threshold = 10

        # Helper functions for tries (prefix trees)
        # TODO (xanda|1-19-17) split out into separate class
        def add_to_trie(idx_list, trie):
            for idx in idx_list[:-1]:
                if idx not in trie:
                    trie[idx] = {}
                trie = trie[idx]
            trie[idx_list[-1]] = 1 + trie.get(idx_list[-1], 0)

        def is_in_trie(idx_list, trie):
            for idx in idx_list:
                if idx not in trie:
                    return False
                trie = trie[idx]
            return trie >= threshold

        # The same tokenizer as the CountVectorizer uses for loading data
        tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')

        # Go through each line and pull out the n-grams, adding them
        # to the trie
        for i, idx in enumerate(self.docs_to_keep):
            if (i + 1) % 1000 == 0:
                print('Line', (i + 1), 'of', len(self.docs_to_keep))
            line = self.doc_lines[idx]
            toks = tokenizer.tokenize(line.lower())
            tok_ids = [self.vocab[tok] for tok in toks if tok in self.vocab]
            if len(tok_ids) < self.max_ngram:
                continue
            for j in range(len(tok_ids) - self.max_ngram + 1):
                sublist = tok_ids[j:j+self.max_ngram]
                add_to_trie(sublist, trie)

        # Go back through each document and delete words that show up in a
        # filtered n-gram. We can't do this greedily because many ngrams will
        # overlap, so we have to collect up the overlaps first.
        inverted_vocab = {v: k for k, v in self.vocab.items()}
        for line_idx, line in enumerate(self.doc_lines):
            if line_idx not in self.docs_to_keep:
                continue
            toks = tokenizer.tokenize(line.lower())
            idx_line = [self.vocab[tok] for tok in toks if tok in self.vocab]
            l = len(idx_line)
            bool_line = [True] * l
            for i in range(l - self.max_ngram + 1):
                if is_in_trie(idx_line[i:i+self.max_ngram], trie):
                    for j in range(i, i+self.max_ngram):
                        bool_line[j] = False
            trimmed_idxs = [idx for idx, keep in zip(idx_line, bool_line) if keep]
            if len(trimmed_idxs) >= self.max_ngram:
                self.doc_lines[line_idx] = (' '.join([inverted_vocab[idx] for idx in trimmed_idxs]))
            else:
                self.docs_to_keep.discard(line_idx)

    def delete_english(self):
        """Deletes lines containing English.
        """
        english_words = ['the', 'and', 'was']
        english_threshold = 8
        english_idxs = [self.vocab[v] for v in english_words]
        english_cts = self.raw_document_vectors[:, english_idxs].sum(axis=1)
        english_idxs, _ = np.where(english_cts > english_threshold)
        self.docs_to_keep.difference_update(english_idxs)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise IndexError("Not enough arguments provided")
    cd = CorpusDeduplicator(
            max_df=0.8,
            diff_tol=0.3,
            max_ngram=7)
    cd.load_corpus_from_file(sys.argv[1])
    if 'reusl' in sys.argv[1]:
        cd.delete_english()
    cd.deduplicate_bow()
    cd.deduplicate_ngrams()
    with open(sys.argv[2], 'w') as f:
        for idx in cd.docs_to_keep:
            f.write(cd.doc_lines[idx] + '\n')

