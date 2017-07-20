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
from plot_helpers import validate_fname
from plot_helpers import make_entity_from_fields

N_PROPS_LIST = [0.0001, 0.001, 0.01, 0.1]

def gather_entropies_for_seq_file(
        output_dir_name,
        seq_file_name,
        is_short):
    # Where we store outputs
    entropy_dict = {}
    entropy_repeated_dict = {}
    entropy_unrepeated_dict = {}

    # Filtering files based on seq file
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = None
        if is_short:
            fields = validate_fname(fname, 'lsa.w.txt', file_prefix=seq_file_prefix)
        else:
            fields = validate_fname(fname, 'lsa.w.txt', file_prefix=seq_file_prefix, n_props_list=N_PROPS_LIST)
        if fields is None:
            continue
        topic_ct = fields['topic_ct']

        # Compute average entropies
        repeats_mask, w_matrix = lsa.read_in_w(os.path.join(output_dir_name, fname))
        entropies = lsa.compute_entropies(w_matrix)
        entropy_dict[topic_ct] = np.mean(entropies)
        entropy_repeated_dict[topic_ct] = np.mean(entropies[repeats_mask])
        if fields['prop'] < 1:
            entropy_unrepeated_dict[topic_ct] = np.mean(entropies[np.logical_not(repeats_mask)])
        else:
            entropy_unrepeated_dict[topic_ct] = 0

    return (entropy_dict, entropy_repeated_dict, entropy_unrepeated_dict)


def plot_exact_entropies(
        input_dir_name,
        output_dir_name,
        file_prefix,
        save_file,
        process=None):
    entropies = []

    for seq_fname in os.listdir(input_dir_name):
        # Check sequence filename is valid
        is_short = file_prefix.endswith('-short')
        seq_fields = None
        if is_short:
            seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=process)
        else:
            seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=process, n_props_list=N_PROPS_LIST)
        if seq_fields is None:
            continue

        entropy_dict, entropy_duplicated, entropy_singular = gather_entropies_for_seq_file(
            output_dir_name,
            seq_fname,
            is_short)
        current_entropies = [
                make_entity_from_fields(
                n_topics, ent, 'all', seq_fields
                ) for n_topics, ent in entropy_dict.items()
            ] + [make_entity_from_fields(
                n_topics, ent, 'duplicated', seq_fields
            ) for n_topics, ent in entropy_duplicated.items()
            ] + [make_entity_from_fields(
                n_topics, ent, 'unduplicated', seq_fields
            ) for n_topics, ent in entropy_singular.items()]
        # add prefixes
        for entity in current_entropies:
          entity['prefix'] = ('lorem' if 'lorem' in output_dir_name else 'sample')
        entropies += current_entropies

    #plt.figure(figsize=(50, 30))
    #dataf = DataFrame([e for e in entropies])
    #g = sns.FacetGrid(
    #        dataf,
    #        col='n_topics',
    #        row='label',
    #        legend_out='true')
    #g.map(sns.pointplot, 'proportion', 'entropy')
    #g.set_titles(col_template="{col_name} topics", row_template="{row_name}")
    #g.fig.subplots_adjust(right=.9)
    #plt.savefig(save_file)

    plt.figure(figsize=(50,30))
    if not entropies:
      raise ValueError("No entropies in list")
    dataf = DataFrame([e for e in entropies])
    g = sns.factorplot(
            x='n_topics',
            y='value',
            hue='proportion',
            col='label',
            row='prefix',
            capsize=.2,
            markers='.',
            scale=0.5,
            data=dataf)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("topic count", "entropy")
    plt.savefig(save_file)


if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    file_prefix = sys.argv[3]
    save_file = sys.argv[4]
    plot_exact_entropies(in_dir, out_dir, file_prefix, save_file)

