#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
import os
import pickle

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

import lda_metrics
from plot_helpers import to_print_name
from plot_helpers import make_entity_from_fields
from plot_helpers import validate_fname



def compute_perplexities_for_seq_file(
        output_dir_name,
        seq_file_name,
        n_tokens,
        repeats_mask,
        n_topics=None):
    # Where we store outputs
    perplexity_repeated_dict = {}
    perplexity_unrepeated_dict = {}

    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = validate_fname(fname, 'traindocprobs', file_prefix=seq_file_prefix)
        if fields is None:
            continue

        # Filter out irrelevant files
        topic_ct = fields['topic_ct']
        if n_topics is not None and topic_ct != n_topics:
            continue

        # Compute average entropies
        prob_file_name = os.path.join(output_dir_name, fname)
        perplexity_repeated_dict[topic_ct] = lda_metrics.compute_perplexity(
                prob_file_name,
                n_tokens,
                repeats_mask)
        perplexity_unrepeated_dict[topic_ct] = lda_metrics.compute_perplexity(
                prob_file_name,
                n_tokens,
                np.logical_not(repeats_mask))

    return perplexity_repeated_dict, perplexity_unrepeated_dict


def plot_exact_perplexities(
        input_dir_names,
        output_dir_names,
        file_prefixes,
        save_file,
        process=None):

    perplexities = []
    pickle_filename = save_file + '.pkl'
    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as pf:
            perplexities = pickle.load(pf)
    else:
        for input_dir_name, output_dir_name, file_prefix in zip(
                input_dir_names,
                output_dir_names,
                file_prefixes):
            for seq_fname in os.listdir(input_dir_name):
                # Check sequence filename is valid
                seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=process)
                if seq_fields is None:
                    continue
    
                # Extract the sequence file info
                (
                    repeated_documents,
                    n_tokens,
                    repeats_mask,
                    doc_models,
                    vocab
                ) = lda_metrics.split_docs_by_repeated(
                            os.path.join(input_dir_name, seq_fname))
    
    
                (perplexity_duplicated, perplexity_singular) = compute_perplexities_for_seq_file(
                    output_dir_name,
                    seq_fname,
                    n_tokens,
                    repeats_mask)
    
                current_perplexities = [
                    make_entity_from_fields(
                        n_topics, per, 'duplicated', seq_fields
                        ) for n_topics, per in perplexity_duplicated.items()
                    ] + [make_entity_from_fields(
                        n_topics, per, 'unduplicated', seq_fields
                        ) for n_topics, per in perplexity_singular.items()
                    ]
                for entity in current_perplexities:
                    entity['prefix'] = ('Lorem Ipsum' if 'lorem' in output_dir_name else 'Sample Article')
                perplexities += current_perplexities
        with open(pickle_filename, 'wb') as pf:
            pickle.dump(perplexities, pf)

    plot_figure_7(perplexities, save_file, value_name='Perplexity')


def plot_figure_7(entities, save_file, value_name="value"):
    if not entities:
        raise ValueError("No entities in list")
    for e in entities:
        if e['label'] == 'unduplicated':
            e['label'] = 'singular'
    dataf = DataFrame([e for e in entities])
    sns.set(style='whitegrid', context='paper', font='serif', font_scale=1.2)
    g = sns.factorplot(
            x='k',
            y='value',
            ylim=(0, None),
            hue='proportion',
            col='label',
            row='prefix',
            capsize=.2,
            markers=['.','o','v','^','<', 's', 'p', 'x'],
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
    g.set_axis_labels("topic count", '')
    plt.savefig(save_file, bbox_inches='tight')


if __name__ == '__main__':
    input_dir_names = [
        # '../lorem_ipsum/long/input',
        '../lorem_ipsum/long/input',
        # '../sample_template/long/input',
        '../sample_template/long/input',
    ]
    output_dir_names = [
        # '../lorem_ipsum/long/output',
        '../lorem_ipsum/long/output',
        # '../sample_template/long/output',
        '../sample_template/long/output',
    ]
    file_prefixes = [
            'reusl-train',
            # 'nyt-train',
            'reusl-train',
            # 'nyt-train',
    ]
    save_file = '../figs/figure_7.pdf'
    plot_exact_perplexities(input_dir_names, output_dir_names, file_prefixes, save_file)
