#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
import random


def sample_lines(lines, proportion):
    """Sample a proportion of the lines to use for modification later.
    
    Arguments:
        lines: list of strings of documents
        proportion: float between 0 and 1 of proportion of documents to use
            in the sample

    Returns:
        a tuple of lists of strings (sampled_lines, unsampled_lines)
    """
    lines_to_sample = set(random.sample(range(len(lines)), int(len(lines) * proportion)))

    sampled_lines = [(i, line) for i, line in enumerate(lines) if i in lines_to_sample]
    unsampled_lines = [(i, line) for i, line in enumerate(lines) if i not in lines_to_sample]
    return sampled_lines, unsampled_lines


def duplicate_exact_lines(
        input_filename,
        proportion_list,
        frequency_list,
        output_filename_format):
    """Create Mallet files with exactly duplicated documents."""
    with open(input_filename) as f:
        lines = [line for line in f]

    for prop in proportion_list:
        sampled_lines, unsampled_lines = sample_lines(lines, prop)
        for freq in frequency_list:
            output_filename = output_filename_format.format(prop, freq)
            write_repeat_file(
                    output_filename,
                    sampled_lines * freq,
                    unsampled_lines)


def duplicate_single_template(
        input_filename,
        proportion_list,
        template_string,
        output_filename_format):
    """Create Mallet files with a prefatory string on documents."""
    with open(input_filename) as f:
        lines = [line for line in f]

    for prop in proportion_list:
        sampled_lines, unsampled_lines = sample_lines(lines, prop)
        sampled_lines = [(i, template_string + ' ' + line) for (i, line) in sampled_lines]
        output_filename = output_filename_format.format(prop)
        write_repeat_file(
                output_filename,
                sampled_lines,
                unsampled_lines)


def duplicate_multiple_templates(
        input_filename,
        proportion_list,
        templates_list,
        output_file_format):
    """Create Mallet files with multiple prefatory strings on documents."""
    with open(input_filename) as f:
        lines = [line for line in f]

    n_templates = len(templates_list)
    for prop in proportion_list:
        sampled_lines, unsampled_lines = sample_lines(lines, prop)
        sampled_lines = [
                (i, templates_list[idx % n_templates] + ' ' + line)
                for idx, (i, line) in enumerate(lines)]
        output_filename = output_file_format.format(prop)
        write_repeat_file(
                output_filename,
                sampled_lines,
                unsampled_lines)


def write_repeat_file(output_filename, treated_lines, untreated_lines):
    """Rewrite file with one line per document into a Mallet-friendly format.
    
    Arguments:
        output_filename: the path to the output file, with the tab-separated
            Mallet format and one line per document.
        treated_lines: lines that had some repeat treatment applied to them.
        untreated_lines: lines that had no treatment applied to them.
    """
    prefix = output_filename.split('.')[0]
    all_lines = [(i, True, line) for i, line in treated_lines] + [(i, False, line) for i, line in untreated_lines]
    random.shuffle(all_lines)
    with open(output_filename, 'w') as f:
        for idx, (i, is_affected, line) in enumerate(all_lines):
            f.write('{}-{}-{}\t{}\t{}'.format(prefix, i, idx, is_affected, line))
