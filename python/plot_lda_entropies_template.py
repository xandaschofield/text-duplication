#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

import lda_metrics


n_props_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
n_freqs_list = [1, 2, 4, 8, 16, 32]
n_topics_list = [5, 10, 20, 40, 80]


def gather_entropies_for_template_seq_file(
        output_dir_name,
        seq_file_name,
        repeats_mask,
        n_topics=None):
    # Where we store outputs
    entropy_dict = {}
    entropy_repeated_dict = {}
    entropy_unrepeated_dict = {}

    # Filtering files based on seq file
    doctopic_fname_regex = r'[a-z\-]+(\d+)-([\d.]+)-(\d+).doctopics'
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        if not fname.startswith(seq_file_prefix + '-'):
            continue

        # Get the fields out of the file name
        doc_topic_match = re.match(doctopic_fname_regex, fname)
        if doc_topic_match is None:
            continue
        f_proc_id, f_prop, f_n_topics = doc_topic_match.group(1, 2, 3)
        f_proc_id = int(f_proc_id)
        f_prop = float(f_prop)
        f_n_topics = int(f_n_topics)

        # Filter out irrelevant files
        if n_topics is not None and f_n_topics != n_topics:
            continue

        # Compute average entropies
        topic_weights_matrix = read_in_doc_topics(os.path.join(output_dir_name, fname))
        entropies = lda_metrics.compute_entropies(topic_weights_matrix)
        entropy_dict[f_n_topics] = np.mean(entropies)
        entropy_repeated_dict[f_n_topics] = np.mean(entropies[repeats_mask])
        if f_prop < 1:
            entropy_unrepeated_dict[f_n_topics] = np.mean(entropies[np.logical_not(repeats_mask)])
        else:
            entropy_unrepeated_dict[f_n_topics] = 0

    return entropy_dict, entropy_repeated_dict, entropy_unrepeated_dict


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


def plot_template_entropies(
        input_dir_name,
        output_dir_name,
        file_prefix,
        save_file,
        n_proc=None):
    entropies = []

    for seq_fname in os.listdir(input_dir_name):
        # Check sequence filename is valid
        if not seq_fname.endswith('.txt'):
            continue
        elif not seq_fname.startswith(file_prefix):
            continue
        elif n_proc is not None and not seq_fname.startswith('{}-{}-'.format(file_prefix, n_proc)):
            continue

        # Pull out the proportion and frequency of repeats from the filename
        seq_fname_regex = r'[a-z\-]+(\d+)-([\d.]+).txt'
        seq_file_match = re.match(seq_fname_regex, seq_fname)
        if seq_file_match is None:
            continue
        s_prop = float(seq_file_match.group(2))

        # Extract the sequence file info
        (
            repeated_documents,
            n_tokens,
            repeats_mask,
            doc_models,
            vocab) = lda_metrics.split_docs_by_repeated(
                    os.path.join(input_dir_name, seq_fname))


        entropy_dict, entropy_duplicated, entropy_singular = gather_entropies_for_template_seq_file(
            output_dir_name,
            seq_fname,
            repeats_mask)
        for n_topics, ent in entropy_dict.items():
            entropies.append({
                'proportion': s_prop,
                'n_topics': n_topics,
                'process_id': n_proc,
                'entropy': ent,
                'contents': 'all',
            })
        for n_topics, ent in entropy_duplicated.items():
            entropies.append({
                'proportion': s_prop,
                'n_topics': n_topics,
                'process_id': n_proc,
                'entropy': ent,
                'contents': 'duplicated',
            })
        for n_topics, ent in entropy_singular.items():
            entropies.append({
                'proportion': s_prop,
                'n_topics': n_topics,
                'process_id': n_proc,
                'entropy': ent,
                'contents': 'unduplicated',
            })

    plt.figure(figsize=(50, 30))
    dataf = DataFrame([e for e in entropies])
    g = sns.FacetGrid(
            dataf,
            col='n_topics',
            row='contents',
            legend_out='true')
    g.map(sns.pointplot, 'proportion', 'entropy')
    g.set_titles(col_template="{col_name} topics", row_template="{row_name}")
    g.fig.subplots_adjust(right=.9)
    plt.savefig(save_file)


if __name__ == '__main__':
    output_dir_name = sys.argv[1]
    input_dir_name = output_dir_name.replace('output', 'input')
    file_prefix = sys.argv[2]
    save_file = sys.argv[3]
    plot_template_entropies(input_dir_name, output_dir_name, file_prefix, save_file)
