#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
from collections import Counter
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


n_props_list = [0.0001, 0.001, 0.01, 0.1]
n_freqs_list = [1, 2, 4, 8, 16]
n_topics_list = [5, 10, 20, 40, 80]

def gather_top_keys_for_seq_file(
        output_dir_name,
        seq_file_name,
        doc_models,
        vocab,
        n_topics=None):
    # Where we store outputs
    top_keys_all_dict = {}
    top_keys_per_dict = {}

    # Filtering files based on seq file
    keyfile_name_regex = r'[a-z\-]+(\d+)-([\d.]+)-(\d+)-(\d+).keys'
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        if not fname.startswith(seq_file_prefix + '-'):
            continue

        # Get the fields out of the file name
        keyfile_match = re.match(keyfile_name_regex, fname)
        if keyfile_match is None:
            continue
        f_proc_id, f_prop, f_freq, f_n_topics = keyfile_match.group(1, 2, 3, 4)
        f_proc_id = int(f_proc_id)
        f_prop = float(f_prop)
        f_freq = int(f_freq)
        f_n_topics = int(f_n_topics)

        # Filter out irrelevant files
        if n_topics is not None and f_n_topics != n_topics:
            continue

        # Compute average top_keys
        keyword_list = read_in_keys(os.path.join(output_dir_name, fname), vocab)
        all_duplicates_set = set(generate_unigrams(doc_models.sum(axis=0)))
        per_duplicate_set = set()
        block_size = 1000
        for block_idx in range(0, doc_models.shape[0], block_size):
            per_duplicate_set.update(generate_unigrams(doc_models[
                block_idx:min(block_idx+block_size, doc_models.shape[0])].todense()))
        top_keys_all_dict[f_n_topics] = float(len([k for k in keyword_list if k in all_duplicates_set])) / len(keyword_list)
        top_keys_per_dict[f_n_topics] = float(len([k for k in keyword_list if k in per_duplicate_set])) / len(keyword_list)

    return top_keys_all_dict, top_keys_per_dict


def generate_unigrams(unigram_vecs):
    top_inds = np.argpartition(unigram_vecs, -20, axis=-1)[:,-20:]
    return set([int(i) for i in np.nditer(top_inds)])

def read_in_keys(keyfile_name, vocab):
    keyword_list = []
    with open(keyfile_name) as keyfile:
        for line in keyfile:
            words = line.strip().split()[2:]
            keyword_list += [vocab.get(w, -1) for w in words]
    return keyword_list


def plot_exact_top_keys(
        input_dir_name,
        output_dir_name,
        file_prefix,
        save_file,
        n_proc=None):
    top_keys = []

    for seq_fname in os.listdir(input_dir_name):
        # Check sequence filename is valid
        if not seq_fname.endswith('.txt'):
            continue
        elif not seq_fname.startswith(file_prefix):
            continue
        elif n_proc is not None and not seq_fname.startswith('{}-{}-'.format(file_prefix, n_proc)):
            continue

        # Pull out the proportion and frequency of repeats from the filename
        seq_fname_regex = r'[a-z\-]+(\d+)-([\d.]+)-(\d+).txt'
        seq_file_match = re.match(seq_fname_regex, seq_fname)
        if seq_file_match is None:
            continue
        s_prop, s_freq = seq_file_match.group(2, 3)
        s_prop = float(s_prop)
        s_freq = int(s_freq)
        if s_freq not in n_freqs_list:
            continue
        if s_prop not in n_props_list:
            continue
        print(seq_fname)

        # Extract the sequence file info
        (
            repeated_documents,
            n_tokens,
            repeats_mask,
            doc_models,
            vocab) = lda_metrics.split_docs_by_repeated(
                    os.path.join(input_dir_name, seq_fname))


        top_keys_all_dict, top_keys_per_dict = gather_top_keys_for_seq_file(
            output_dir_name,
            seq_fname,
            doc_models,
            vocab)
        for n_topics, prop in top_keys_all_dict.items():
            top_keys.append({
                'proportion': s_prop,
                'frequency': s_freq,
                'n_topics': n_topics,
                'process_id': n_proc,
                'top_keys': prop,
                'contents': 'all',
            })
        for n_topics, prop in top_keys_per_dict.items():
            top_keys.append({
                'proportion': s_prop,
                'frequency': s_freq,
                'n_topics': n_topics,
                'process_id': n_proc,
                'top_keys': prop,
                'contents': 'per duplicate',
            })

    plt.figure(figsize=(50, 20))
    dataf = DataFrame([e for e in top_keys])
    g = sns.FacetGrid(
            dataf,
            col='n_topics',
            row='contents')
    cbar_ax = g.fig.add_axes([.92, .3, .02, .4])
    g.map_dataframe(facet_heatmap, cbar_ax=cbar_ax)
    g.set_titles(col_template="{col_name} topics", row_template="{row_name}")
    g.fig.subplots_adjust(right=.9)
    plt.savefig(save_file)


def facet_heatmap(data, color, **kws):
    data = data.pivot(index='proportion', columns='frequency', values='top_keys')
    sns.heatmap(data, cmap='Blues', vmin=0, vmax=1, **kws)


if __name__ == '__main__':
    input_dir_name = '../exact_duplicates_input'
    output_dir_name = '../exact_duplicates_output'
    file_prefix = sys.argv[1]
    save_file = sys.argv[2]
    plot_exact_top_keys(input_dir_name, output_dir_name, file_prefix, save_file)
