#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield and Laure Thompson
import os
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
import scipy.stats
import seaborn as sns

import lsa
from plot_helpers import make_entity_from_fields
from plot_helpers import validate_fname

N_FREQS_LIST = [2**x for x in range(15)]

def gather_loss_for_seq_file(
        output_dir_name,
        seq_file_name):
    # Where we store outputs
    loss_dict = {}
    loss_repeated_dict = {}
    loss_unrepeated_dict = {}

    # Filtering files based on seq file
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = validate_fname(fname, 'lsa.loss.txt', file_prefix=seq_file_prefix, n_freqs_list=N_FREQS_LIST)
        if fields is None:
            continue
        topic_ct = fields['topic_ct']

        # Compute average loss
        repeats_mask, loss = lsa.read_in_loss(os.path.join(output_dir_name, fname))
        loss_dict[topic_ct] = np.mean(loss)
        loss_repeated_dict[topic_ct] = np.mean(loss[repeats_mask])
        loss_unrepeated_dict[topic_ct] = np.mean(loss[np.logical_not(repeats_mask)])

    return (loss_dict, loss_repeated_dict, loss_unrepeated_dict)


def plot_exact_loss(
        input_dir_name,
        output_dir_name,
        file_prefix,
        save_file,
        n_proc=None):
    loss = []

    for seq_fname in os.listdir(input_dir_name):
        # Check sequence filename is valid
        seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=n_proc, n_freqs_list=N_FREQS_LIST)
        if seq_fields is None:
            continue

        (
            loss_dict,
            loss_duplicated,
            loss_singular
        ) = gather_loss_for_seq_file(
            output_dir_name,
            seq_fname)

        loss += [make_entity_from_fields(
                n_topics, l, 'all', seq_fields
            ) for n_topics, l in loss_dict.items()]
        loss += [make_entity_from_fields(
                n_topics, l, 'duplicated', seq_fields
            ) for n_topics, l in loss_duplicated.items()]
        loss += [make_entity_from_fields(
                n_topics, l, 'unduplicated', seq_fields
            ) for n_topics, l in loss_singular.items()]

    plt.figure(figsize=(50, 30))
    dataf = DataFrame([l for l in loss])
    g = sns.FacetGrid(
            dataf,
            col='n_topics',
            row='label',
            legend_out='true')
    g.map(sns.pointplot, 'frequency', 'value')
    g.set_titles(col_template="{col_name} topics", row_template="{row_name}")
    g.set_xticklabels(rotation=90)
    plt.savefig(save_file)


if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    file_prefix = sys.argv[3]
    save_file = sys.argv[4]
    plot_exact_loss(in_dir, out_dir, file_prefix, save_file)
