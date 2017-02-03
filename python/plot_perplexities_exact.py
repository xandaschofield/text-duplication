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
n_freqs_list = [1, 2, 4, 8, 16]
n_topics_list = [5, 10, 20, 40, 80]

def compute_perplexities_for_seq_file(
        output_dir_name,
        seq_file_name,
        n_tokens,
        repeats_mask,
        n_topics=None):
    # Where we store outputs
    perplexity_dict = {}
    perplexity_repeated_dict = {}
    perplexity_unrepeated_dict = {}

    # Filtering files based on seq file
    perplexity_fname_regex = r'[a-z\-]+(\d+)-([\d.]+)-(\d+)-(\d+).traindocprobs'
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        if not fname.startswith(seq_file_prefix + '-'):
            continue

        # Get the fields out of the file name
        perplexity_match = re.match(perplexity_fname_regex, fname)
        if perplexity_match is None:
            continue
        f_proc_id, f_prop, f_freq, f_n_topics = perplexity_match.group(1, 2, 3, 4)
        f_proc_id = int(f_proc_id)
        f_prop = float(f_prop)
        f_freq = int(f_freq)
        f_n_topics = int(f_n_topics)

        # Filter out irrelevant files
        if n_topics is not None and f_n_topics != n_topics:
            continue

        # Compute average entropies
        prob_file_name = os.path.join(output_dir_name, fname)
        perplexity_dict[f_n_topics] = lda_metrics.compute_perplexity(
                prob_file_name,
                n_tokens)
        perplexity_repeated_dict[f_n_topics] = lda_metrics.compute_perplexity(
                prob_file_name,
                n_tokens,
                repeats_mask)
        if f_prop < 1:
            perplexity_unrepeated_dict[f_n_topics] = lda_metrics.compute_perplexity(
                prob_file_name,
                n_tokens,
                np.logical_not(repeats_mask))
        else:
            perplexity_unrepeated_dict[f_n_topics] = 0

    return perplexity_dict, perplexity_repeated_dict, perplexity_unrepeated_dict


def plot_exact_perplexities(
        input_dir_name,
        output_dir_name,
        file_prefix,
        save_file,
        n_proc=None):
    perplexities = []

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
        print(seq_fname)

        # Extract the sequence file info
        (
            repeated_documents,
            n_tokens,
            repeats_mask,
            doc_models,
            vocab) = lda_metrics.split_docs_by_repeated(
                    os.path.join(input_dir_name, seq_fname))

        (
            perplexity_dict,
            perplexity_duplicated,
            perplexity_singular
        ) = compute_perplexities_for_seq_file(
            output_dir_name,
            seq_fname,
            n_tokens,
            repeats_mask)

        for n_topics, per in perplexity_dict.items():
            perplexities.append({
                'proportion': s_prop,
                'frequency': s_freq,
                'n_topics': n_topics,
                'process_id': n_proc,
                'perplexity': per,
                'contents': 'all',
            })
        for n_topics, per in perplexity_duplicated.items():
            perplexities.append({
                'proportion': s_prop,
                'frequency': s_freq,
                'n_topics': n_topics,
                'process_id': n_proc,
                'perplexity': per,
                'contents': 'duplicated',
            })
        for n_topics, per in perplexity_singular.items():
            perplexities.append({
                'proportion': s_prop,
                'frequency': s_freq,
                'n_topics': n_topics,
                'process_id': n_proc,
                'perplexity': per,
                'contents': 'unduplicated',
            })

    plt.figure(figsize=(50, 30))
    dataf = DataFrame([p for p in perplexities if p['frequency'] == 8])
    g = sns.FacetGrid(
            dataf,
            col='n_topics',
            row='contents')
    cbar_ax = g.fig.add_axes([.92, .3, .02, .4])
    g.map_dataframe(facet_heatmap, cbar_ax=cbar_ax)
    g.set_titles(col_template="{col_name} topics", row_template="{row_name}")
    g.fig.subplots_adjust(right=.9)
    # Dict structure: dict[prop][freq] = list[entropies]
    # For each seq file, get entropies, update a dict
    # Turn this into a color plot
    plt.savefig(save_file)


def facet_heatmap(data, color, **kws):
    data = data.pivot(index='proportion', columns='frequency', values='perplexity')
    sns.heatmap(data, cmap='Blues', **kws)


if __name__ == '__main__':
    input_dir_name = '../exact_duplicates_input'
    output_dir_name = '../exact_duplicates_output'
    file_prefix = sys.argv[1]
    save_file = sys.argv[2]
    plot_exact_perplexities(input_dir_name, output_dir_name, file_prefix, save_file)
