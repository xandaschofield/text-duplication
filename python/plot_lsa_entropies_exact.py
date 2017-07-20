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
from plot_helpers import print_significances
from plot_helpers import validate_fname

N_PROPS_LIST = [0.0001, 0.001, 0.01, 0.1]

def gather_entropies_for_seq_file(
        output_dir_name,
        seq_file_name,
        is_short):
    # Where we store outputs
    entropy_dict = {}
    entropy_repeated_dict = {}
    entropy_unrepeated_dict = {}

    # Filtering files based on seq file
    seq_file_prefix = seq_file_name[:-4]
    for fname in os.listdir(output_dir_name):
        fields = None
        if is_short:
            fields = validate_fname(fname, 'lsa.w.txt', file_prefix=seq_file_prefix)
        else:
            fields = validate_fname(fname, 'lsa.w.txt', file_prefix=seq_file_prefix, n_props_list=N_PROPS_LIST)
        if fields is None:
            continue
        topic_ct = fields['topic_ct']

        # Compute average entropies
        repeats_mask, w_matrix = lsa.read_in_w(os.path.join(output_dir_name, fname))
        entropies = lsa.compute_entropies(w_matrix)
        entropy_dict[topic_ct] = np.mean(entropies)
        entropy_repeated_dict[topic_ct] = np.mean(entropies[repeats_mask])
        if fields['prop'] < 1:
            entropy_unrepeated_dict[topic_ct] = np.mean(entropies[np.logical_not(repeats_mask)])
        else:
            entropy_unrepeated_dict[topic_ct] = 0

    return (entropy_dict, entropy_repeated_dict, entropy_unrepeated_dict)


def plot_exact_entropies(
        input_dir_name,
        output_dir_name,
        file_prefix,
        save_file,
        process=None):
    entropies = []

    for seq_fname in os.listdir(input_dir_name):
        # Check sequence filename is valiid
        is_short = file_prefix.endswith('-short')
        seq_fields = None
        if is_short:
            seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=process)
        else:
            seq_fields = validate_fname(seq_fname, 'txt', file_prefix=file_prefix, process=process, n_props_list=N_PROPS_LIST)
        if seq_fields is None:
            continue

        entropy_dict, entropy_duplicated, entropy_singular = gather_entropies_for_seq_file(
            output_dir_name,
            seq_fname,
            is_short)
        entropies += [make_entity_from_fields(
                n_topics, ent, 'all', seq_fields
            ) for n_topics, ent in entropy_dict.items()]
        entropies += [make_entity_from_fields(
                n_topics, ent, 'duplicated', seq_fields
            ) for n_topics, ent in entropy_duplicated.items()]
        entropies += [make_entity_from_fields(
                n_topics, ent, 'unduplicated', seq_fields
            ) for n_topics, ent in entropy_singular.items()]

    #plot_cmap_from_entity_list(entropies, save_file, vmax=6.0)
    plot_pplot_from_entity_list(entropies, save_file, value_name='Entropy')
    #print_significances(entropies)

if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    file_prefix = sys.argv[3]
    save_file = sys.argv[4]
    plot_exact_entropies(in_dir, out_dir, file_prefix, save_file)

