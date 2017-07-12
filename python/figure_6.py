#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
import os
import pickle

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns

import lda_metrics
from plot_helpers import to_print_name
from plot_helpers import make_entity_from_fields
from plot_helpers import validate_fname

sns.set_style(style='whitegrid')


def compute_perplexities_for_seq_file(
        output_dir_name,
        seq_file_name,
        n_tokens,
        n_topics=None):
    # Where we store outputs
    perplexity_dict = {}

    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = validate_fname(fname, 'testdocprobs', file_prefix=seq_file_prefix)
        if fields is None:
            continue
        
        # Filter out irrelevant files
        topic_ct = fields['topic_ct']
        if n_topics is not None and topic_ct != n_topics:
            continue

        # Compute average entropies
        prob_file_name = os.path.join(output_dir_name, fname)
        perplexity_dict[topic_ct] = float(lda_metrics.compute_perplexity(
                prob_file_name,
                n_tokens,
                None))

    return perplexity_dict


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
    
            if 'reusl' in input_dir_name:
                test_seq_name = '../data/reusl-test.txt'
            else:
                test_seq_name = '../data/nyt-test.txt'
    
            # Extract the sequence file info
            (
                _,
                _,
                _,
                _,
                vocab
            ) = lda_metrics.split_docs_by_repeated(os.path.join(
                input_dir_name,
                os.listdir(input_dir_name)[0]))
            n_tokens = lda_metrics.get_test_doc_counts(test_seq_name, vocab=vocab)
    
    
            for seq_fname in os.listdir(input_dir_name):
                # Check sequence filename is valid
                seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=process)
                if seq_fields is None:
                    continue
    
                perplexity_dict = compute_perplexities_for_seq_file(
                    output_dir_name,
                    seq_fname,
                    n_tokens)
    
                current_perplexities = [
                    make_entity_from_fields(
                        n_topics, per, 'test', seq_fields
                        ) for n_topics, per in perplexity_dict.items()
                    ]
                for entity in current_perplexities:
                    entity['prefix'] = to_print_name[file_prefix]
                perplexities += current_perplexities
        with open(pickle_filename, 'wb') as pf:
            pickle.dump(perplexities, pf)

    plot_figure_6(perplexities, save_file, value_name='Perplexity')


def plot_figure_6(entities, save_file, value_name="value"):
    if not entities:
        raise ValueError("No entities in list")
    for e in entities:
        if e['label'] == 'unduplicated':
            e['label'] = 'singular'
        e['value'] = e['value'] / 1000
    dataf = DataFrame([e for e in entities])
    sns.set(style='whitegrid', context='notebook', font='serif', font_scale=1.2)
    g = sns.factorplot(
            x='c',
            y='value',
            hue='proportion',
            col='k',
            row='prefix',
            capsize=.1,
            markers=['.','o','v','^','<', 's', 'p', 'x'],
            sharex=False,
            sharey=False,
            aspect=1.2,
            size=2.5,
            scale=0.6,
            errwidth=1,
            dodge=0.2,
            legend_out=True,
            data=dataf)
    g.despine(left=True)
    g.set_titles('{row_name}, {col_name} topics')
    g.set_axis_labels("# copies", '')
    plt.savefig(save_file, bbox_inches='tight')

def facet_heatmap(data, color, **kws):
    data = data.pivot_table(index='proportion', columns='frequency', values='value')
    sns.heatmap(data, cmap='Blues', annot=True, fmt=".2f", **kws)

if __name__ == '__main__':
    input_dir_names = [
#            '../exact_duplicates/long/input',
#            '../exact_duplicates/short/input',
            '../exact_duplicates/long/input',
#            '../exact_duplicates/short/input'
    ]
    output_dir_names = [
#            '../exact_duplicates/long/output',
#            '../exact_duplicates/short/output',
            '../exact_duplicates/long/output',
#            '../exact_duplicates/short/output'
    ]
    file_prefixes = [
#            'reusl-train',
#            'reusl-short',
            'nyt-train',
#            'nyt-short'
    ]
    save_file = '../figs/figure_6.pdf'
    plot_exact_perplexities(input_dir_names, output_dir_names, file_prefixes, save_file)
