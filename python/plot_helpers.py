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


n_props_list = ['None', 0.001, 0.01, 0.1]
n_freqs_list = [1, 2, 4, 8]
n_topics_list = [5, 10, 20, 40, 80, 160, 320]


def validate_fname(
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
        'frequency': fields.get('freq', 0),
        'n_topics': n_topics,
        'process_id': fields['proc_id'],
        'value': val,
        'label': label
    }


def plot_cmap_from_entity_list(entities, save_file, vmax=1.0):
    plt.figure(figsize=(25, 15))
    dataf = DataFrame([e for e in entities])
    g = sns.FacetGrid(
            dataf,
            col='n_topics',
            row='label')
    cbar_ax = g.fig.add_axes([.92, .3, .02, .4])
    g.map_dataframe(facet_heatmap, cbar_ax=cbar_ax, vmax=vmax)
    g.set_titles(col_template="{col_name} topics", row_template="{row_name}")
    g.fig.subplots_adjust(right=.9)
    plt.savefig(save_file)


def plot_pplot_from_entity_list(entities, save_file):
    plt.figure(figsize=(25, 15))
    dataf = DataFrame([e for e in entities])
    g = sns.factorplot(
            x='proportion',
            y='value',
            hue='frequency',
            col='n_topics',
            row='label',
            capsize=.2,
            markers='.',
            data=dataf)
    g.set_titles(col_template="{col_name} topics", row_template="{row_name}")
    plt.savefig(save_file)


def facet_heatmap(data, color, vmax=1.0, **kws):
    data = data.pivot_table(index='proportion', columns='frequency', values='value')
    sns.heatmap(data, cmap='Blues', annot=True, fmt=".2f", vmin=0, vmax=vmax, **kws)
