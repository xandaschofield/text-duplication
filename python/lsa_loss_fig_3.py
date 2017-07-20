#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Laure Thompson
import os
import pickle

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

import lsa
from plot_lsa_loss_template import gather_loss_for_seq_file
from plot_helpers import to_print_name
from plot_helpers import make_entity_from_fields
from plot_helpers import validate_fname


N_PROPS_LIST = [0.0001, 0.001, 0.01, 0.1]

def plot_figure(
        in_dir_names,
        out_dir_names,
        file_prefixes,
        save_file,
        process=None):

    losses = []
    pickle_filename = save_file + '.pkl'
    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as pf:
            losses = pickle.load(pf)
    else:
        for in_dir_name, out_dir_name, file_prefix in zip(
                in_dir_names,
                out_dir_names,
                file_prefixes):
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

                loss_dict, loss_duplicated, loss_singular = gather_loss_for_seq_file(
                    out_dir_name,
                    seq_fname,
                    is_short)

                current_losses = [
                    make_entity_from_fields(
                        n_topics, l, 'duplicated', seq_fields
                        ) for n_topics, l in loss_duplicated.items()
                    ] + [make_entity_from_fields(
                        n_topics, l, 'unduplicated', seq_fields
                    ) for n_topics, l, in loss_singular.items()
                    ]
                for loss in current_losses:
                    loss['prefix'] = ('Lorem Ipsum' if 'lorem' in out_dir_name else 'Sample Article')
                losses += current_losses
        with open(pickle_filename, 'wb') as pf:
            pickle.dump(losses, pf)

    # Plot Figure
    if not losses:
        raise ValueError("No losses in list")
    for e in losses:
        if e['label'] == 'unduplicated':
            e['label'] = 'untemplated'
        elif e['label'] == 'duplicated':
            e['label'] = 'templated'
    dataf = DataFrame([l for l in losses])
    sns.set(style='whitegrid', context='paper', font='serif', font_scale=1.3)
    g = sns.factorplot(
            x='K',
            y='value',
            ylim=(0, 1),
            hue='proportion',
            row='prefix',
            col='label',
            capsize=.2,
            markers=['.','o','v','^'],
            aspect=1.9,
            size=2,
            scale=0.6,
            errwidth=1,
            legend=False,
            dodge=0.2,
            sharex=False,
            data=dataf)
    g.despine(left=True)
    g.axes[0][1].legend(loc='lower left')
    g.set_titles('{row_name}, {col_name}')
    #g.set_axis_labels("Proportion", "Loss")
    g.set_axis_labels("# components", '')
    
    plt.savefig(save_file, bbox_inches='tight')



if __name__ == '__main__':
#    short_in_dir_names = [
#            '../lorem_ipsum/short/input',
#            '../sample_template/short/input'
#    ]
#    short_out_dir_names = [
#            '/home/ljt82/text-duplication/lorem_ipsum/short/output',
#            '/home/ljt82/text-duplication/sample_template/short/output'
#    ]
#    short_file_prefixes = [
#            'reusl-short',
#            'reusl-short'
#    ]
#    short_save_file = 'lsa_loss_fig_3-short_nyt.pdf'
    
#    plot_figure(
#            short_in_dir_names, short_out_dir_names, 
#            short_file_prefixes, short_save_file
#    )

#    print('Finished plotting short')

    long_in_dir_names = [
            '../lorem_ipsum/long/input',
            '../sample_template/long/input'
    ]
    long_out_dir_names = [
            '/home/ljt82/text-duplication/lorem_ipsum/long/output',
            '/home/ljt82/text-duplication/sample_template/long/output'
    ]
    long_file_prefixes = [
            'reusl-train',
            'reusl-train'
    ]
    long_save_file = 'lsa_loss_fig_3-long_new.pdf'
    plot_figure(
            long_in_dir_names, long_out_dir_names, 
            long_file_prefixes, long_save_file
    )


