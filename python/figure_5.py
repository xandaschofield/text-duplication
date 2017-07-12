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
from plot_helpers import facet_heatmap
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
        fields = validate_fname(fname, 'traindocprobs', file_prefix=seq_file_prefix, n_freqs_list=[2**i for i in range(15)])
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
                seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=process, n_freqs_list=[2**i for i in range(15)])
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
                        n_topics, per, 'unduplicated', seq_fields
                        ) for n_topics, per in perplexity_singular.items()
                    ]
                for entity in current_perplexities:
                    entity['prefix'] = to_print_name[file_prefix]
                perplexities += current_perplexities
        with open(pickle_filename, 'wb') as pf:
            pickle.dump(perplexities, pf)

    plot_figure_5(perplexities, save_file, value_name='Perplexity')


def plot_figure_5(entities, save_file, value_name="value"):
    if not entities:
        raise ValueError("No entities in list")
    for e in entities:
        if e['label'] == 'unduplicated':
            e['label'] = 'singular'
        e['K'] = e['k']
    dataf = DataFrame([e for e in entities])
    sns.set(style='whitegrid', context='paper', font='serif', font_scale=1.2)
    g = sns.factorplot(
            x='c',
            y='value',
            hue='K',
            col='prefix',
            capsize=.2,
            markers=['.','o','v','^','<', 's', 'p', 'x'],
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
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.set_axis_labels("# copies", '')
    plt.savefig(save_file, bbox_inches='tight')


if __name__ == '__main__':
    input_dir_names = [
#            '../exact_duplicates_singledoc/long/input',
            '../exact_duplicates_singledoc/short/input',
#            '../exact_duplicates_singledoc/long/input',
            '../exact_duplicates_singledoc/short/input'
    ]
    output_dir_names = [
#            '../exact_duplicates_singledoc/long/output',
            '../exact_duplicates_singledoc/short/output',
#            '../exact_duplicates_singledoc/long/output',
            '../exact_duplicates_singledoc/short/output'
    ]
    file_prefixes = [
#            'reusl-train',
            'reusl-short',
#            'nyt-train',
            'nyt-short'
    ]
    save_file = '../figs/figure_5.pdf'
    plot_exact_perplexities(input_dir_names, output_dir_names, file_prefixes, save_file)
