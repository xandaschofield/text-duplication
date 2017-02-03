#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
import os
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns

import lda_metrics


n_props_list = ['None', 0.0001, 0.001, 0.01, 0.1]
n_freqs_list = [1, 2, 4, 8]
n_topics_list = [5, 10, 20, 40, 80, 160, 320]

def check_fname_valid(
    fname,
    extension,
    file_prefix=None,
    process=None,
    n_topics=None):
    if not fname.startswith(file_prefix + '-'):
        return None

    is_seq_file = (extension == 'txt')
    is_exact_duplicate = (len(fname.split('-')) == 6 - int(is_seq_file))
    if is_exact_duplicate:
        if is_seq_file:
            fname_regex = r'[a-z\-]+(?P<proc_id>\d+)-(?P<prop>[\d.]+|None)-(?P<freq>\d+).' + extension
        else:
            fname_regex = r'[a-z\-]+(?P<proc_id>\d+)-(?P<prop>[\d.]+|None)-(?P<freq>\d+)-(?P<topic_ct>\d+).' + extension
    else:
        if is_seq_file:
            fname_regex = r'[a-z\-]+(?P<proc_id>\d+)-(?P<prop>[\d.]+|None).' + extension
        else:
            fname_regex = r'[a-z\-]+(?P<proc_id>\d+)-(?P<prop>[\d.]+|None)-(?P<topic_ct>\d+).' + extension

    match_obj = re.match(fname_regex, fname)
    if match_obj is None:
        return None
    ret_dict = {}

    proc_id = int(match_obj.group('proc_id'))
    if process is not None and proc_id != process:
        return None
    else:
        ret_dict['proc_id'] = proc_id

    prop = match_obj.group('prop')
    if prop is not 'None':
        prop = float(prop)
    if prop not in n_props_list:
        return None
    else:
        ret_dict['prop'] = prop

    if is_seq_file:
        topic_ct = int(match_obj.group('topic_ct'))
        if not (n_topics is None) and topic_ct != n_topics:
            return None
        elif not (n_topics in n_topics_list):
            return None
        else:
            ret_dict['topic_ct'] = topic_ct

    if is_exact_duplicate:
        freq = int(match_obj.group('freq'))
        if freq not in n_freqs_list:
            return None
        else:
            ret_dict['freq'] = freq

    return ret_dict


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
        fields = check_fname_valid(fname, '.doctopics', seq_file_prefix=seq_file_prefix)
        if fields is None:
            continue
        topic_ct = fields['topic_ct']

        # Compute average entropies
        topic_weights_matrix = read_in_doc_topics(os.path.join(output_dir_name, fname))
        entropies = lda_metrics.compute_entropies(topic_weights_matrix)
        entropy_dict[topic_ct] = np.mean(entropies)
        entropy_repeated_dict[topic_ct] = np.mean(entropies[repeats_mask])
        if fields['prop'] < 1:
            entropy_unrepeated_dict[topic_ct] = np.mean(entropies[np.logical_not(repeats_mask)])
        else:
            entropy_unrepeated_dict[topic_ct] = 0

    return entropy_dict, entropy_repeated_dict, entropy_unrepeated_dict


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


def make_entity_from_fields(n_topics, val, label, fields):
    return {
        'proportion': fields['prop'],
        'frequency': fields.get('freq', 0),
        'n_topics': n_topics,
        'process_id': fields['proc_id'],
        'value': val,
        'label': label
    }


def plot_exact_entropies(
        input_dir_name,
        output_dir_name,
        file_prefix,
        save_file,
        n_proc=None):
    entropies = []

    for seq_fname in os.listdir(input_dir_name):
        # Check sequence filename is valid
        seq_fields = check_fname_valid(seq_fname, 'txt', file_prefix=file_prefix, n_proc=n_proc)

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
        entropies += [make_entity_from_fields(
                n_topics, ent, 'all', seq_fields
            ) for n_topics, ent in entropy_dict.items()]
        entropies += [make_entity_from_fields(
                n_topics, ent, 'duplicated', seq_fields
            ) for n_topics, ent in entropy_duplicated.items()]
        entropies += [make_entity_from_fields(
                n_topics, ent, 'unduplicated', seq_fields
            ) for n_topics, ent in entropy_singular.items()]

    plot_cmap_from_entity_list(entropies, save_file)

def plot_cmap_from_entity_list(entities, save_file):
    plt.figure(figsize=(50, 30))
    dataf = DataFrame([e for e in entities])
    g = sns.FacetGrid(
            dataf,
            col='n_topics',
            row='label')
    cbar_ax = g.fig.add_axes([.92, .3, .02, .4])
    g.map_dataframe(facet_heatmap, cbar_ax=cbar_ax)
    g.set_titles(col_template="{col_name} topics", row_template="{row_name}")
    g.fig.subplots_adjust(right=.9)
    plt.savefig(save_file)


def facet_heatmap(data, color, **kws):
    data = data.pivot(index='proportion', columns='frequency', values='value')
    sns.heatmap(data, cmap='Blues', **kws)
