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


def gather_entropies_for_seq_file(
        output_dir_name,
        seq_file_name,
        repeats_mask,
        n_topics=80):
    # Where we store outputs
    entropy_dict = {}
    entropy_repeated_dict = {}
    entropy_unrepeated_dict = {}

    # Filtering files based on seq file
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = validate_fname(fname, 'doctopics', file_prefix=seq_file_prefix)
        if fields is None:
            continue
        topic_ct = fields['topic_ct']
        if n_topics is not None and topic_ct != n_topics:
            continue
        # Compute average entropies
        try:
            topic_weights_matrix = read_in_doc_topics(os.path.join(output_dir_name, fname))
            entropies = lda_metrics.compute_entropies(topic_weights_matrix)
            entropy_dict[topic_ct] = np.mean(entropies)
            entropy_repeated_dict[topic_ct] = np.mean(entropies[repeats_mask])
            if fields['prop'] < 1:
                entropy_unrepeated_dict[topic_ct] = np.mean(entropies[np.logical_not(repeats_mask)])
            else:
                entropy_unrepeated_dict[topic_ct] = 0
        except:
            continue

    return (entropy_dict, entropy_repeated_dict, entropy_unrepeated_dict)


def read_in_doc_topics(doc_topic_fname):
    topic_weights_list = []
    with open(doc_topic_fname) as doc_topic_file:
        for i, line in enumerate(doc_topic_file):
            row = line.strip().split('\t')
            _, original_id, line_id = row[1].split('-')
            topic_weights = np.array([float(c) for c in row[2:]])
            topic_weights_list.append(topic_weights)
    topic_weights_matrix = np.vstack(topic_weights_list)
    return topic_weights_matrix


def plot_exact_entropies(
        input_dir_names,
        output_dir_names,
        file_prefixes,
        save_file,
        fixed_topic_no=80,
        process=None):
    entropies = []

    pickle_filename = save_file + '.pkl'
    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as pf:
            entropies = pickle.load(pf)

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

                entropy_dict, entropy_duplicated, entropy_singular = gather_entropies_for_seq_file(
                    output_dir_name,
                    seq_fname,
                    repeats_mask,
                    n_topics=fixed_topic_no)
                current_entropies = [
                        make_entity_from_fields(
                        n_topics, ent, 'duplicated', seq_fields
                        ) for n_topics, ent in entropy_duplicated.items()
                        if n_topics == fixed_topic_no
                    ] + [make_entity_from_fields(
                        n_topics, ent, 'singular', seq_fields
                    ) for n_topics, ent in entropy_singular.items()
                        if n_topics == fixed_topic_no
                    ]
                for entity in current_entropies:
                    entity['prefix'] = to_print_name[file_prefix]
                entropies += current_entropies

        with open(pickle_filename, 'wb') as pf:
            pickle.dump(entropies, pf)
    # plot_cmap_from_entity_list(entropies, save_file, vmax=3.0)
    plot_figure_1(entropies, save_file, value_name='Entropy')


def plot_figure_1(entities, save_file, value_name="value"):
    if not entities:
        raise ValueError("No entities in list")
    dataf = DataFrame([e for e in entities])
    sns.set(style='whitegrid', context='notebook', font='serif', font_scale=1.2)
    g = sns.factorplot(
            x='c',
            y='value',
            hue='proportion',
            col='label',
            row='prefix',
            capsize=.1,
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
    g.axes[0][0].legend()
    g.set_titles('{row_name}, {col_name}')
    g.set_axis_labels("# copies", '')
    plt.savefig(save_file, bbox_inches='tight')



if __name__ == '__main__':
    input_dir_names = [
            '../exact_duplicates/long/input',
#            '../exact_duplicates/short/input',
            '../exact_duplicates/long/input',
#            '../exact_duplicates/short/input'
    ]
    output_dir_names = [
            '../exact_duplicates/long/output',
#            '../exact_duplicates/short/output',
            '../exact_duplicates/long/output',
#            '../exact_duplicates/short/output'
    ]
    file_prefixes = [
            'reusl-train',
#            'reusl-short',
            'nyt-train',
#            'nyt-short'
    ]
    save_file = '../figs/figure_1a.pdf'
    plot_exact_entropies(input_dir_names, output_dir_names, file_prefixes, save_file)
    input_dir_names = [
#            '../exact_duplicates/long/input',
            '../exact_duplicates/short/input',
#            '../exact_duplicates/long/input',
            '../exact_duplicates/short/input'
    ]
    output_dir_names = [
#            '../exact_duplicates/long/output',
            '../exact_duplicates/short/output',
#            '../exact_duplicates/long/output',
            '../exact_duplicates/short/output'
    ]
    file_prefixes = [
#            'reusl-train',
            'reusl-short',
#            'nyt-train',
            'nyt-short'
    ]
    save_file = '../figs/figure_1b.pdf'
    plot_exact_entropies(input_dir_names, output_dir_names, file_prefixes, save_file)

