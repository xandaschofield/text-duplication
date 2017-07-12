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
        repeats_mask):
    # Where we store outputs
    entropy_dict = {}
    entropy_repeated_dict = {}
    entropy_unrepeated_dict = {}

    # Filtering files based on seq file
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = validate_fname(
                fname,
                'doctopics',
                file_prefix=seq_file_prefix,
                n_freqs_list=[2**i for i in range(15)]
        )
        if fields is None:
            continue
        topic_ct = fields['topic_ct']

        # Compute average entropies
        try:
            topic_weights_matrix = read_in_doc_topics(os.path.join(output_dir_name, fname))
            entropies = lda_metrics.compute_entropies(topic_weights_matrix)
            entropy_dict[topic_ct] = np.mean(entropies)
            entropy_repeated_dict[topic_ct] = np.mean(entropies[repeats_mask])
            entropy_unrepeated_dict[topic_ct] = np.mean(entropies[np.logical_not(repeats_mask)])
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
                seq_fields = validate_fname(
                        seq_fname,
                        'txt',
                        file_prefix=file_prefix,
                        process=process,
                        n_freqs_list=[2**i for i in range(15)]
                )
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
                    repeats_mask)
                #entropies += [make_entity_from_fields(
                #        n_topics, ent, 'all', seq_fields
                #    ) for n_topics, ent in entropy_dict.items()]
                current_entropies = [
                        make_entity_from_fields(
                        n_topics, ent, 'unduplicated', seq_fields
                    ) for n_topics, ent in entropy_singular.items()
                    ]
                for entity in current_entropies:
                    entity['prefix'] = to_print_name[file_prefix]
                entropies += current_entropies
        with open(pickle_filename, 'wb') as pf:
            pickle.dump(entropies, pf)

    # plot_cmap_from_entity_list(entropies, save_file, vmax=3.0)
    plot_figure_2(entropies, save_file, value_name='Entropy')


def plot_figure_2(entities, save_file, value_name="value"):
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
    g.set_titles("{col_name}")
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
    save_file = '../figs/figure_2.pdf'
    plot_exact_entropies(input_dir_names, output_dir_names, file_prefixes, save_file)

