#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
from collections import defaultdict
import os
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from pandas import DataFrame
import scipy.stats
import seaborn as sns

import lda_metrics


N_PROPS_LIST = ['None', 0.001, 0.01, 0.1]
N_FREQS_LIST = [1, 2, 4, 8]
N_TOPICS_LIST = [5, 10, 20, 40, 80, 160, 320]


sns.set(style='whitegrid', context='poster')


to_print_name = {
        'reusl-train': 'REUSL 25k',
        'reusl-short': 'REUSL 2.5k',
        'nyt-train': 'NYT 25k',
        'nyt-short': 'NYT 2.5k',
        }


def validate_fname(
    fname,
    extension,
    file_prefix=None,
    process=None,
    n_topics=None,
    n_props_list=N_PROPS_LIST,
    n_freqs_list=N_FREQS_LIST,
    n_topics_list=N_TOPICS_LIST):
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
    if prop != 'None':
        prop = float(prop)
    if prop not in n_props_list:
        return None
    else:
        ret_dict['prop'] = prop

    if not is_seq_file:
        topic_ct = int(match_obj.group('topic_ct'))
        if not (n_topics is None) and topic_ct != n_topics:
            return None
        elif not (topic_ct in n_topics_list):
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


def make_entity_from_fields(n_topics, val, label, fields):
    return {
        'proportion': fields['prop'],
        'c': fields.get('freq', 0),
        'K': n_topics,
        'process_id': fields['proc_id'],
        'value': val,
        'label': label
    }


def print_significances(entities):
    val_collection = defaultdict(list)
    for entity in entities:
        key = "{} {} {} {}".format(
                entity['label'],
                entity['proportion'],
                entity['k'],
                entity['c'])
        val_collection[key].append(entity['value'])
    for key in sorted(val_collection.keys()):
        print(key, np.mean(val_collection[key]), 1.96*scipy.stats.sem(val_collection[key]))


def plot_cmap_from_entity_list(entities, save_file, vmax=1.0, value_name="value"):
    plt.figure(figsize=(25, 15))
    if not entities:
        raise ValueError("No entities in list")
    dataf = DataFrame([e for e in entities])
    g = sns.FacetGrid(
            dataf,
            col='k',
            row='label')
    cbar_ax = g.fig.add_axes([.92, .3, .02, .4])
    g.map_dataframe(facet_heatmap, cbar_ax=cbar_ax, vmax=vmax)
    g.set_titles(col_template="{col_name} topics", row_template="{row_name}")
    g.fig.subplots_adjust(right=.9)
    plt.savefig(save_file)


def plot_pplot_from_entity_list(entities, save_file, value_name="value"):
    plt.figure(figsize=(25, 15))
    if not entities:
        raise ValueError("No entities in list")
    dataf = DataFrame([e for e in entities])
    g = sns.factorplot(
            x='c',
            y='value',
            hue='proportion',
            col='k',
            row='label',
            capsize=.2,
            markers='.',
            scale=0.5,
            data=dataf)
    g.set_titles(col_template="{col_name} topics", row_template="{row_name}")
    g.set_axis_labels("# copies", value_name)
    plt.savefig(save_file)


def print_data_table(entities):
    dataf = DataFrame([e for e in entities])
    data = dataf.pivot_table(index='proportion', columns='c', values='value')
    print(data)


def facet_heatmap(data, color, vmax=1.0, **kws):
    data = data.pivot_table(index='proportion', columns='c', values='value')
    sns.heatmap(data, cmap='Blues', annot=True, fmt=".2f", vmin=0, vmax=vmax, **kws)
