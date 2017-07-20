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

#N_PROPS_LIST = [0.0001, 0.001, 0.01, 0.1]
N_PROPS_LIST = [0.001, 0.01, 0.1]

def gather_loss_for_seq_file(
      output_dir_name,
      seq_file_name,
      is_short,
      fixed_topic_ct):
    # Where we store outputs
    loss_dict = {} 
    loss_repeated_dict = {}
    loss_unrepeated_dict = {}

    # Filtering files based on seq file
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = None
        if is_short:
            fields = validate_fname(fname, 'lsa.loss.txt', file_prefix=seq_file_prefix)
        else:
            fields = validate_fname(fname, 'lsa.loss.txt', file_prefix=seq_file_prefix, n_props_list=N_PROPS_LIST)
        if fields is None:
            continue
        topic_ct = fields['topic_ct']
        if topic_ct != fixed_topic_ct:
            continue

        # Compute average loss
        repeats_mask, loss = lsa.read_in_loss(os.path.join(output_dir_name, fname))
        loss_dict[topic_ct] = np.mean(loss)
        loss_repeated_dict[topic_ct] = np.mean(loss[repeats_mask])
        loss_unrepeated_dict[topic_ct] = np.mean(loss[np.logical_not(repeats_mask)])

    return (loss_dict, loss_repeated_dict, loss_unrepeated_dict)


def plot_figure(
        in_dir_names,
        out_dir_names,
        file_prefixes,
        save_file,
        fixed_topic_no=80,
        process=None):
    losses = []

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
            loss_dict, loss_duplicated, loss_singular = gather_loss_for_seq_file(
                out_dir_name,
                seq_fname,
                is_short,
                fixed_topic_no)

            current_losses = [
                    make_entity_from_fields(
                    n_topics, l, 'duplicated', seq_fields
                    ) for n_topics, l in loss_duplicated.items()
                    if n_topics == fixed_topic_no
                ] + [make_entity_from_fields(
                    n_topics, l, 'unduplicated', seq_fields
                ) for n_topics, l, in loss_singular.items()
                    if n_topics == fixed_topic_no
                ]
            for loss in current_losses:
                loss['prefix'] = to_print_name[file_prefix]
            losses += current_losses

    # Plot Figure
    if not losses:
        raise ValueError("No losses in list")
    for l in losses:
        if l['label'] == 'unduplicated':
              l['label'] = 'singular'
    dataf = DataFrame([l for l in losses])
    sns.set(style='whitegrid', context='paper', font='serif', font_scale=1.2)
    g = sns.factorplot(
            x='c',
            y='value',
            hue='proportion',
            col='label',
            row='prefix',
            capsize=.1,
            markers=['.','o','v','^','<','s','p','x'],
            sharex=False,
            sharey=False,
            aspect=1.5,
            size=2,
            scale=0.6,
            errorwidth=1,
            dodge=0.2,
            legend_out=True,
            data=dataf)
    g.despine(left=True)
    g.set_titles('{row_name}, {col_name}')
    g.set_axis_labels("# copies", '')
    plt.savefig(save_file, bbox_inches='tight')


if __name__ == '__main__':
#    short_in_dir_names = [
#            '../exact_duplicates/short/input',
#            '../exact_duplicates/short/input'
#    ]
#    short_out_dir_names = [
#            '/home/ljt82/text-duplication/exact_duplicates/short/output',
#            '/home/ljt82/text-duplication/exact_duplicates/short/output'
#    ]
#    short_file_prefixes = [
#            'reusl-short',
#            'nyt-short'
#    ]
#    short_save_file = 'lsa_loss_fig_1-short.pdf'
#    
#    plot_figure(
#            short_in_dir_names, short_out_dir_names, 
#            short_file_prefixes, short_save_file
#    )
#
#    print('Finished plotting short')

    long_in_dir_names = [
            '../exact_duplicates/long/input',
            '../exact_duplicates/long/input'
    ]
    long_out_dir_names = [
            '/home/ljt82/text-duplication/exact_duplicates/long/output',
            '/home/ljt82/text-duplication/exact_duplicates/long/output'
    ]
    long_file_prefixes = [
            'reusl-train',
            'nyt-train'
    ]
    long_save_file = 'lsa_loss_fig_1-long.pdf'
    plot_figure(
            long_in_dir_names, long_out_dir_names, 
            long_file_prefixes, long_save_file
    )


