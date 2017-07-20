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
from plot_lsa_entropies_template import gather_entropies_for_seq_file
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
                is_short)

            current_entropies = [
                    make_entity_from_fields(
                    n_topics, ent, 'duplicated', seq_fields
                    ) for n_topics, ent in entropy_duplicated.items()
                ] + [make_entity_from_fields(
                    n_topics, ent, 'unduplicated', seq_fields
                ) for n_topics, ent, in entropy_singular.items()
                ]
            for entropy in current_entropies:
                entropy['prefix'] = ('Lorem Ipsum' if 'lorem' in out_dir_name else 'Sample Article')
            entropies += current_entropies
    # Plot Figure
    if not entropies:
        raise ValueError("No entropies in list")
    dataf = DataFrame([e for e in entropies])
    g = sns.factorplot(
            x='n_topics',
            y='value',
            #ylim(0,2.5),
            hue='proportion',
            col='label',
            row='prefix',
            capsize=.2,
            markers='.',
            scale=0.5,
            data=dataf)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("# of components", "Entropy")
    plt.savefig(save_file)



if __name__ == '__main__':
    short_in_dir_names = [
            '../lorem_ipsum/short/input',
            '../sample_template/short/input'
    ]
    short_out_dir_names = [
            '/home/ljt82/text-duplication/lorem_ipsum/short/output',
            '/home/ljt82/text-duplication/sample_template/short/output'
    ]
    short_file_prefixes = [
            'reusl-short',
            'reusl-short'
    ]
    short_save_file = 'lsa_fig_3-short.pdf'
    
    plot_figure(
            short_in_dir_names, short_out_dir_names, 
            short_file_prefixes, short_save_file
    )

    print('Finished plotting short')

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
    long_save_file = 'lsa_fig_3-long.pdf'
    plot_figure(
            long_in_dir_names, long_out_dir_names, 
            long_file_prefixes, long_save_file
    )


