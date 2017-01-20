#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield

"""corpus_deduplicator.py

An class for deduplicating corpora.
"""
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction import text


class CorpusDeduplicator(object):

    def __init__(self,
            token_pattern=r"\w[\w\-']+\w",
            max_df=0.8,
            diff_tol=0.2,
            max_ngram=10
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

    def load_corpus_from_file_as_unigrams(self, filename):
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
        self.raw_document_vectors = corpus_cv.fit_transform(self.doc_lines)
        self.vocab = corpus_cv.vocabulary_
        
    def deduplicate_by_unigrams(self):
        """Returns a set of document indices to retain based upon removing
        texts with high unigram overlap, retaining only the longest in each
        overlap check.

        Returns:
        a set of integers identifying indices of documents to retain

        """
        doc_lengths = self.raw_document_vectors.sum(axis=1)
        n_docs = len(doc_lengths)
        docs_to_keep = set(range(n_docs))

    def deduplicate_ngrams(self):
        """Modifies the lines of text to remove long n-grams that appear
        frequently. Edits self.doc_lines in place.
        """
        trie = {}
        dup_trie = {}
        idx_lines = []

        # Helper functions for tries (prefix trees)
        # TODO (xanda|1-19-17) split out into separate class
        def add_to_trie(idx_list, trie):
            for idx in idx_list[:-1]:
                if idx not in trie:
                    trie[idx] = {}
                trie = trie[idx]
            trie[idx_list[-1]] = 1 + trie.get(idx_list[-1], 0)
            return trie[idx_list[-1]]

        def is_in_trie(idx_list, trie):
            for idx in idx_list:
                if idx not in trie:
                    return False
                trie = trie[idx]
            return True

        # The same tokenizer as the CountVectorizer uses for loading data
        tokenizer = RegexpTokenizer(r'(?u)\b\w\w+\b')

        # Go through each line and pull out the n-grams, adding them
        # to the trie
        # TODO (xanda|1-19-17) this is a little too memory hungry right now
        for i, line in enumerate(self.doc_lines):
            if i % 1000 == 0:
                print('Line', i)
            toks = tokenizer.tokenize(line.lower())
            tok_ids = [self.vocab[tok] for tok in toks if tok in self.vocab]
            if len(tok_ids) < self.max_ngram:
                continue
            idx_lines.append(tok_ids)
            for j in range(len(tok_ids) - self.max_ngram + 1):
                sublist = tok_ids[j:j+self.max_ngram]
                ct = add_to_trie(sublist, trie)
                # For those that have shown up enough, add them to the
                # official trie for deduplication
                if ct > 20:
                    add_to_trie(sublist, dup_trie)

        # Go back through each document and delete words that show up in a
        # filtered n-gram. We can't do this greedily because many ngrams will
        # overlap, so we have to collect up the overlaps first.
        new_lines = []
        inverted_vocab = {v: k for k, v in self.vocab.items()}
        for idx_line in idx_lines:
            l = len(idx_line)
            bool_line = [True] * l
            for i in range(l + self.max_ngram - 1):
                if is_in_trie(idx_line[i:i+self.max_ngram], dup_trie):
                    for j in range(i, i+self.max_ngram):
                        bool_line[j] = False
            trimmed_idxs = [idx for idx, keep in zip(idx_line, bool_line) if keep]
            if len(trimmed_idxs) >= self.max_ngram:
                new_lines.append(' '.join([inverted_vocab[idx] for idx in trimmed_idxs]))

        self.doc_lines = new_lines
        return
