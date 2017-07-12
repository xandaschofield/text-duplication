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


def gather_keys_for_seq_file(
        output_dir_name,
        seq_file_name,
        doc_models,
        vocab,
        n_topics=80):
    # Where we store outputs
    top_keys_all_dict = {} 

    # Filtering files based on seq file
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = validate_fname(fname, 'keys', file_prefix=seq_file_prefix)
        if fields is None:
            continue
        topic_ct = fields['topic_ct']
        if n_topics is not None and topic_ct != n_topics:
            continue
        try:
            keyword_list = read_in_keys(os.path.join(output_dir_name, fname), vocab)
            all_duplicates_set = set(generate_unigrams(doc_models.sum(axis=0)))
        except:
            continue
        top_keys_all_dict[topic_ct] = float(len([k for k in keyword_list if k in all_duplicates_set])) / len(keyword_list)

    return top_keys_all_dict


def generate_unigrams(unigram_vecs):
    top_inds = np.argpartition(unigram_vecs, -20, axis=-1)[:,-20:]
    return set([int(i) for i in np.nditer(top_inds)])

def read_in_keys(keyfile_name, vocab):
    keyword_list = []
    with open(keyfile_name) as keyfile:
        for line in keyfile:
            words = line.strip().split()[2:]
            keyword_list += [vocab.get(w, -1) for w in words]
    return keyword_list



def plot_exact_top_keys(
        input_dir_names,
        output_dir_names,
        file_prefixes,
        save_file,
        fixed_topic_no=80,
        process=None):
    key_cts = []

    pickle_filename = save_file + '.pkl'
    if os.path.exists(pickle_filename):
        with open(pickle_filename, 'rb') as pf:
            key_cts = pickle.load(pf)

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
    
                top_keys_dict = gather_keys_for_seq_file(
                    output_dir_name,
                    seq_fname,
                    doc_models,
                    vocab,
                    n_topics=fixed_topic_no)
                #entropies += [make_entity_from_fields(
                #        n_topics, ent, 'all', seq_fields
                #    ) for n_topics, ent in entropy_dict.items()]
                current_key_cts = [make_entity_from_fields(
                        n_topics, kct, 'unduplicated', seq_fields
                    ) for n_topics, kct in top_keys_dict.items()
                        if n_topics == fixed_topic_no
                    ]
                for entity in current_key_cts:
                    entity['prefix'] = to_print_name[file_prefix]
                key_cts += current_key_cts
        with open(pickle_filename, 'wb') as pf:
            pickle.dump(key_cts, pf)

    # plot_cmap_from_entity_list(entropies, save_file, vmax=3.0)
    plot_figure_8(key_cts, save_file, value_name='Proportion of keys')


def plot_figure_8(entities, save_file, value_name="value"):
    if not entities:
        raise ValueError("No entities in list")
    dataf = DataFrame([e for e in entities])
    sns.set(style='whitegrid', context='notebook', font='serif', font_scale=1.2)
    g = sns.factorplot(
            x='c',
            y='value',
            ylim=(0, 1),
            hue='proportion',
            col='prefix',
            capsize=.1,
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
    g.set_titles('{col_name}')
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
    save_file = 'figure_8.pdf'
    plot_exact_top_keys(input_dir_names, output_dir_names, file_prefixes, save_file)

