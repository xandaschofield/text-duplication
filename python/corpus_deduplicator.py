#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield

"""corpus_deduplicator.py

An class for deduplicating corpora.
"""
import numpy
from sklearn.feature_extraction import text


class CorpusDeduplicator(object):

    def __init__(self,
            token_pattern=r"\w[\w\-']+\w",
            max_df=0.8,
            diff_tol=0.2,
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
        self.diff_tol = diff_tol

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
            doc_lines = [line for line in corpus_file]
        raw_document_vectors = corpus_cv.fit_transform(doc_lines)
        return raw_document_vectors
        
    def deduplicate_by_unigrams(self, raw_document_vectors):
        """Returns a set of document indices to retain based upon removing
        texts with high unigram overlap, retaining only the longest in each
        overlap check.

        Arguments:
            raw_document_vectors::numpy matrix -- a matrix document-word counts
       
        Returns:
            a set of integers identifying indices of documents to retain

        """
        doc_lengths = raw_document_vectors.sum(axis=1)
        n_docs = len(doc_lengths)
        docs_to_keep = set(range(n_docs))

        for current_idx in range(n_docs):
            if current_idx not in docs_to_keep:
                continue
            doc_indices = list(docs_to_keep)
            doc_diffs = numpy.clip(
                    (
                        raw_document_vectors[doc_indices]
                        - raw_document_vectors[current_idx].toarray().ravel()
                    ),
                    0,
                    None
            ).sum(axis=1) / doc_lengths
            similar_docs = [
                    idx for idx, diff in zip(doc_indices, doc_diffs)
                    if diff < self.diff_tol]
            
            docs_to_keep -= set(similar_docs)
            docs_to_keep.add(similar_docs[
                numpy.argmax(doc_lengths[similar_docs])
            ])

        return docs_to_keep
