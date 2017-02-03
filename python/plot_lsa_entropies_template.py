#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield and Laure Thompson
import os
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

import lsa


n_props_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
n_freqs_list = [1, 2, 4, 8, 16, 32]
n_topics_list = [5, 10, 20, 40, 80]


def gather_entropies_for_template_txt_file(
        output_dir_name,
        txt_file_name,
        n_topics=None):
    # Where we store outputs
    entropy_dict = {}
    entropy_repeated_dict = {}
    entropy_unrepeated_dict = {}

    # Filtering files based on txt file
    w_fname_regex = r'[a-z\-]+(\d+)-([\d.]+)-(\d+).lsa-w.txt'
    txt_file_prefix = txt_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        if not fname.startswith(txt_file_prefix + '-'):
            continue

        # Get the fields out of the file name
        w_match = re.match(w_fname_regex, fname)
        if w_match is None:
            continue
        f_proc_id, f_prop, f_n_topics = w_match.group(1, 2, 3)
        f_proc_id = int(f_proc_id)
        f_prop = float(f_prop)
        f_n_topics = int(f_n_topics)

        # Filter out irrelevant files
        if n_topics is not None and f_n_topics != n_topics:
            continue

        # Compute average entropies
        repeats_mask, w_matrix = read_in_w(os.path.join(output_dir_name, fname))
        print('{}'.format(w_matrix.shape), flush=True)
        entropies = lsa.compute_entropies(w_matrix)
        entropy_dict[f_n_topics] = np.mean(entropies)
        entropy_repeated_dict[f_n_topics] = np.mean(entropies[repeats_mask])
        if f_prop < 1:
            entropy_unrepeated_dict[f_n_topics] = np.mean(entropies[np.logical_not(repeats_mask)])
        else:
            entropy_unrepeated_dict[f_n_topics] = 0

    return entropy_dict, entropy_repeated_dict, entropy_unrepeated_dict


def read_in_w(w_fname):
    repeats_mask = []
    w_rows_list = []
    with open(w_fname) as w_file:
        for i, line in enumerate(w_file):
            row = line.strip().split(' ')
            _, original_id, line_id = row[1].split('-')
            label = (row[2] == 'True')
            repeats_mask.append(label)
            w_row = np.array([float(c) for c in row[3:]])
            w_rows_list.append(w_row)
    repeats_mask = np.array(repeats_mask, dtype=bool)
    w_matrix = np.vstack(w_rows_list)
    return repeats_mask, w_matrix


def plot_template_entropies(
        input_dir_name,
        output_dir_name,
        file_prefix,
        save_file,
        n_proc=None):
    entropies = []

    for txt_fname in os.listdir(input_dir_name):
        # Check text filename is valid
        if not txt_fname.endswith('.txt'):
            continue
        elif not txt_fname.startswith(file_prefix):
            continue
        elif n_proc is not None and not txt_fname.startswith('{}-{}-'.format(file_prefix, n_proc)):
            continue

        # Pull out the proportion and frequency of repeats from the filename
        txt_fname_regex = r'[a-z\-]+(\d+)-([\d.]+).txt'
        txt_file_match = re.match(txt_fname_regex, txt_fname)
        if txt_file_match is None:
            continue
        s_prop = float(txt_file_match.group(2))

        entropy_dict, entropy_duplicated, entropy_singular = gather_entropies_for_template_txt_file(
            output_dir_name,
            txt_fname)
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
    input_dir_name = '../sample_template_input'
    output_dir_name = '../sample_template_output'
    file_prefix = 'reusl-train'
    save_file = 'lsa_sample.png'
    plot_template_entropies(input_dir_name, output_dir_name, file_prefix, save_file)
