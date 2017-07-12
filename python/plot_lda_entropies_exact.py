#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
import os
import sys

import numpy as np

import lda_metrics
from plot_helpers import make_entity_from_fields
from plot_helpers import plot_cmap_from_entity_list 
from plot_helpers import plot_pplot_from_entity_list 
from plot_helpers import validate_fname


def gather_entropies_for_seq_file(
        output_dir_name,
        seq_file_name,
        repeats_mask):
    # Where we store outputs
    entropy_dict = {}
    entropy_repeated_dict = {}
    entropy_unrepeated_dict = {}

    # Filtering files based on seq file
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = validate_fname(fname, 'doctopics', file_prefix=seq_file_prefix)
        if fields is None:
            continue
        topic_ct = fields['topic_ct']

        # Compute average entropies
        topic_weights_matrix = read_in_doc_topics(os.path.join(output_dir_name, fname))
        entropies = lda_metrics.compute_entropies(topic_weights_matrix)
        entropy_dict[topic_ct] = np.mean(entropies)
        entropy_repeated_dict[topic_ct] = np.mean(entropies[repeats_mask])
        if fields['prop'] < 1:
            entropy_unrepeated_dict[topic_ct] = np.mean(entropies[np.logical_not(repeats_mask)])
        else:
            entropy_unrepeated_dict[topic_ct] = 0

    return (entropy_dict, entropy_repeated_dict, entropy_unrepeated_dict)


def read_in_doc_topics(doc_topic_fname):
    topic_weights_list = []
    with open(doc_topic_fname) as doc_topic_file:
        for i, line in enumerate(doc_topic_file):
            row = line.strip().split('\t')
            _, original_id, line_id = row[1].split('-')
            topic_weights = np.array([float(c) for c in row[2:]])
            topic_weights_list.append(topic_weights)
    topic_weights_matrix = np.vstack(topic_weights_list)
    return topic_weights_matrix


def plot_exact_entropies(
        input_dir_name,
        output_dir_name,
        file_prefix,
        save_file,
        process=None):
    entropies = []

    for seq_fname in os.listdir(input_dir_name):
        # Check sequence filename is valid
        seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=process)
        if seq_fields is None:
            continue
        # Extract the sequence file info
        (
            repeated_documents,
            n_tokens,
            repeats_mask,
            doc_models,
            vocab
        ) = lda_metrics.split_docs_by_repeated(
                    os.path.join(input_dir_name, seq_fname))

        entropy_dict, entropy_duplicated, entropy_singular = gather_entropies_for_seq_file(
            output_dir_name,
            seq_fname,
            repeats_mask)
        entropies += [make_entity_from_fields(
                n_topics, ent, 'all', seq_fields
            ) for n_topics, ent in entropy_dict.items()]
        entropies += [make_entity_from_fields(
                n_topics, ent, 'duplicated', seq_fields
            ) for n_topics, ent in entropy_duplicated.items()]
        entropies += [make_entity_from_fields(
                n_topics, ent, 'unduplicated', seq_fields
            ) for n_topics, ent in entropy_singular.items()]

    # plot_cmap_from_entity_list(entropies, save_file, vmax=3.0)
    plot_pplot_from_entity_list(entropies, save_file)


if __name__ == '__main__':
    input_dir_name = '../exact_duplicates/short/input'
    output_dir_name = '../exact_duplicates/short/output'
    file_prefix = sys.argv[1]
    save_file = sys.argv[2]
    plot_exact_entropies(input_dir_name, output_dir_name, file_prefix, save_file)

