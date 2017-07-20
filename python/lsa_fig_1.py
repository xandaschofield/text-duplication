#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Laure Thompson
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

import lsa
from plot_helpers import to_print_name
from plot_helpers import make_entity_from_fields
from plot_helpers import validate_fname

N_PROPS_LIST = [0.0001, 0.001, 0.01, 0.1]

def gather_entropies_for_seq_file(
      output_dir_name,
      seq_file_name,
      is_short,
      fixed_topic_ct):
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
        if topic_ct != fixed_topic_ct:
            continue

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


def plot_figure(
        in_dir_names,
        out_dir_names,
        file_prefixes,
        save_file,
        fixed_topic_no=80,
        process=None):
    entropies = []

    for in_dir_name, out_dir_name, file_prefix in zip(
            in_dir_names,
            out_dir_names,
            file_prefixes):
        print('Current in_dir: {}'.format(in_dir_name))
        for seq_fname in os.listdir(in_dir_name):
            # Check sequence filename is valid
            is_short = file_prefix.endswith('-short')
            seq_fields = None
            if is_short:
                seq_fields = validate_fname(
                    seq_fname, 'txt', 
                    file_prefix=file_prefix,
                    process=process
                )
            else:
                seq_fields = validate_fname(
                    seq_fname, 'txt',
                    file_prefix=file_prefix,
                    process=process,
                    n_props_list=N_PROPS_LIST
                )
            if seq_fields is None:
                continue

            print("seq_fname={}".format(seq_fname))
            entropy_dict, entropy_duplicated, entropy_singular = gather_entropies_for_seq_file(
                out_dir_name,
                seq_fname,
                is_short,
                fixed_topic_no)

            current_entropies = [
                    make_entity_from_fields(
                    n_topics, ent, 'duplicated', seq_fields
                    ) for n_topics, ent in entropy_duplicated.items()
                    if n_topics == fixed_topic_no
                ] + [make_entity_from_fields(
                    n_topics, ent, 'unduplicated', seq_fields
                ) for n_topics, ent, in entropy_singular.items()
                    if n_topics == fixed_topic_no
                ]
            for entropy in current_entropies:
                entropy['prefix'] = to_print_name[file_prefix]
            entropies += current_entropies

    # Plot Figure
    if not entropies:
        raise ValueError("No entropies in list")
    for e in entropies:
      if e['label'] == 'unduplicated':
        e['label'] = 'singular'
    dataf = DataFrame([e for e in entropies])
    sns.set(style='whitegrid', context='paper', font='serif', font_scale=1.2)
    g = sns.factorplot(
            x='c',
            y='value',
            hue='proportion',
            col='label',
            row='prefix',
            capsize=.1,
            markers=['.', 'o', 'v', '^', '<', 's', 'p', 'x'],
            sharex=False,
            sharey=False,
            aspect=1.5,
            size=2,
            scale=0.6,
            errwidth=1,
            dodge=0.2,
            legend=False,
            data=dataf)
    g.despine(left=True)
    g.axes[0][0].legend(loc='best')
    g.set_titles('{row_name}, {col_name}')
    g.set_axis_labels("# copies", '')
    plt.savefig(save_file, bbox_inches='tight')


if __name__ == '__main__':
    short_in_dir_names = [
            '../exact_duplicates/short/input',
            '../exact_duplicates/short/input'
    ]
    short_out_dir_names = [
            '/home/ljt82/text-duplication/exact_duplicates/short/output',
            '/home/ljt82/text-duplication/exact_duplicates/short/output'
    ]
    short_file_prefixes = [
            'reusl-short',
            'nyt-short'
    ]
    short_save_file = 'lsa_fig_1-short.pdf'
    
    plot_figure(
            short_in_dir_names, short_out_dir_names, 
            short_file_prefixes, short_save_file
    )

    print('Finished plotting short')

#    long_in_dir_names = [
#            '../exact_duplicates/long/input',
#            '../exact_duplicates/long/input'
#    ]
#    long_out_dir_names = [
#            '/home/ljt82/text-duplication/exact_duplicates/long/output',
#            '/home/ljt82/text-duplication/exact_duplicates/long/output'
#    ]
#    long_file_prefixes = [
#            'reusl-train',
#            'nyt-train'
#    ]
#    long_save_file = 'lsa_fig_1-long.pdf'
#    plot_figure(
#            long_in_dir_names, long_out_dir_names, 
#            long_file_prefixes, long_save_file
#    )


