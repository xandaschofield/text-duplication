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
from plot_lsa_loss_exact_single import gather_loss_for_seq_file
from plot_helpers import to_print_name
from plot_helpers import make_entity_from_fields
from plot_helpers import validate_fname

N_FREQS_LIST = [2**x for x in range(15)]

def plot_figure(
        in_dir_names,
        out_dir_names,
        file_prefixes,
        save_file,
        process=None):
    losses = []

    for in_dir_name, out_dir_name, file_prefix in zip(
            in_dir_names,
            out_dir_names,
            file_prefixes):
        print('Current in_dir: {}'.format(in_dir_name))
        for seq_fname in os.listdir(in_dir_name):
            # Check sequence filename is valid
            seq_fields = validate_fname(
                    seq_fname, 'txt', 
                    file_prefix=file_prefix,
                    process=process,
                    n_freqs_list=N_FREQS_LIST
                )
            if seq_fields is None:
                continue

            print("seq_fname={}".format(seq_fname))
            loss_dict, loss_duplicated, loss_singular = gather_loss_for_seq_file(
                out_dir_name,
                seq_fname)

            current_losses = [
                    make_entity_from_fields(
                    n_topics, l, 'duplicated', seq_fields
                    ) for n_topics, l in loss_duplicated.items()
                ] + [make_entity_from_fields(
                    n_topics, l, 'unduplicated', seq_fields
                ) for n_topics, l, in loss_singular.items()
                ]
            for loss in current_losses:
                loss['prefix'] = to_print_name[file_prefix]
            losses += current_losses

    # Plot Figure
    if not losses:
        raise ValueError('No losses in list')
    for l in losses:
      if l['label'] == 'unduplicated':
        l['label'] = 'singular'
    dataf = DataFrame([l for l in losses])
    sns.set(style='whitegrid', context='paper', font='serif', font_scale=1.2)
    g = sns.factorplot(
            x='c',
            y='value',
            hue='K',
            col='label',
            row='prefix',
            capsize=.2,
            markers=['.', 'o', 'v', '^', '<', 's', 'p', 'x'],
            sharex=False,
            sharey=False,
            aspect=1.5,
            size=2,
            scale=0.6,
            errwidth=1,
            dodge=0.2,
            legend_out=True,
            data=dataf)
    g.despine(left=True)
    g.set_xticklabels(rotation=60)
    for plot in g.axes[0]:
        for idx, label in enumerate(plot.get_xticklabels()):
            label.set_visible(idx % 2 == 0)
    g.set_titles('{row_name}, {col_name}')
    g.set_axis_labels('# copies', '')
    plt.savefig(save_file, bbox_inches='tight')


if __name__ == '__main__':
    short_in_dir_names = [
            '../exact_duplicates_singledoc/short/input',
            #'../exact_duplicates_singledoc/short/input'
    ]
    short_out_dir_names = [
            '/home/ljt82/text-duplication/exact_duplicates_singledoc/short/output',
            #'/home/ljt82/text-duplication/exact_duplicates_singledoc/short/output'
    ]
    short_file_prefixes = [
            #'reusl-short',
            'nyt-short'
    ]
    short_save_file = 'lsa_loss_fig_2-short.pdf'
    
    plot_figure(
            short_in_dir_names, short_out_dir_names, 
            short_file_prefixes, short_save_file
    )

    print('Finished plotting short')

#    long_in_dir_names = [
#            #'../exact_duplicates_singledoc/long/input',
#            '../exact_duplicates_singledoc/long/input'
#    ]
#    long_out_dir_names = [
#            #'/home/ljt82/text-duplication/exact_duplicates_singledoc/long/output',
#            '/home/ljt82/text-duplication/exact_duplicates_singledoc/long/output'
#    ]
#    long_file_prefixes = [
#            #'reusl-train',
#            'nyt-train'
#    ]
#    long_save_file = 'lsa_loss_fig_2-long.pdf'
#    plot_figure(
#            long_in_dir_names, long_out_dir_names, 
#            long_file_prefixes, long_save_file
#    )



