#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
import sys

from nltk.tokenize import RegexpTokenizer

import duplicate_writing


def get_template_string(filename, n_toks, line_idx=None):
    with open(filename) as template_file:
        lines = [line for line in template_file]
        if line_idx is not None:
            template_line = lines[line_idx]
        else:
            template_line = ' '.join(lines)
    tokenizer = RegexpTokenizer(r"\w[\w\-']+\w")
    toks = tokenizer.tokenize(template_line.lower())
    final_template_string = ' '.join(toks[:n_toks])
    return final_template_string


if __name__ == "__main__":
    proportion_list = [0.0001, 0.001, 0.01, 0.1]
    line_idx = None
    if len(sys.argv) == 5:
        _, input_filename, format_string_file, template_file, n_toks_str = sys.argv
    elif len(sys.argv) == 6:
        _, input_filename, format_string_file, template_file, n_toks_str, line_idx = sys.argv
        line_idx = int(line_idx) % 10
    template_string = get_template_string(
            template_file,
            int(n_toks_str),
            line_idx=line_idx)
    duplicate_writing.duplicate_single_template(
            input_filename,
            proportion_list,
            template_string,
            format_string_file)
