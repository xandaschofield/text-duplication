#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield and Laure Thompson
import os
import sys

import numpy as np

import lsa
from plot_helpers import make_entity_from_fields
from plot_helpers import plot_cmap_from_entity_list
from plot_helpers import plot_pplot_from_entity_list
from plot_helpers import validate_fname

N_PROPS_LIST = [0.0001, 0.001, 0.01, 0.1]

def gather_loss_for_seq_file(
        output_dir_name,
        seq_file_name,
        is_short):
    # Where we store outputs
    loss_dict = {}
    loss_repeated_dict = {}
    loss_unrepeated_dict = {}

    # Filtering files based on seq file
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = None
        if is_short:
            fields = validate_fname(fname, 'lsa.loss.txt', file_prefix=seq_file_prefix)
        else:
            fields = validate_fname(fname, 'lsa.loss.txt', file_prefix=seq_file_prefix, n_props_list=N_PROPS_LIST)
        if fields is None:
            continue
        topic_ct = fields['topic_ct']

        # Compute average loss
        repeats_mask, loss = lsa.read_in_loss(os.path.join(output_dir_name, fname))
        loss_dict[topic_ct] = np.mean(loss)
        loss_repeated_dict[topic_ct] = np.mean(loss[repeats_mask])
        if fields['prop'] < 1:
            loss_unrepeated_dict[topic_ct] = np.mean(loss[np.logical_not(repeats_mask)])
        else:
            loss_unrepeated_dict[topic_ct] = 0

    return (loss_dict, loss_repeated_dict, loss_unrepeated_dict)


def plot_exact_loss(
        input_dir_name,
        output_dir_name,
        file_prefix,
        save_file,
        n_proc=None):
    loss = []

    for seq_fname in os.listdir(input_dir_name):
        # Check sequence filename is valid
        seq_fields = None
        is_short = file_prefix.endswith('-short')
        if is_short:
            seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=n_proc)
        else:
            seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=n_proc, n_props_list=N_PROPS_LIST)
        if seq_fields is None:
            continue

        (
            loss_dict,
            loss_duplicated,
            loss_singular
        ) = gather_loss_for_seq_file(
            output_dir_name,
            seq_fname,
            is_short)

        loss += [make_entity_from_fields(
                n_topics, l, 'all', seq_fields
            ) for n_topics, l in loss_dict.items()]
        loss += [make_entity_from_fields(
                n_topics, l, 'duplicated', seq_fields
            ) for n_topics, l in loss_duplicated.items()]
        loss += [make_entity_from_fields(
                n_topics, l, 'unduplicated', seq_fields
            ) for n_topics, l in loss_singular.items()]

    plot_cmap_from_entity_list(loss, save_file, value_name="Loss")


if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    file_prefix = sys.argv[3]
    save_file = sys.argv[4]
    plot_exact_loss(in_dir, out_dir, file_prefix, save_file)
