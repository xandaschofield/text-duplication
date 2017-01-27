#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
import duplicate_writing


if __name__ == "__main__":
    input_filename = sys.argv[1]
    proportion_list = [0.0001, 0.001, 0.01, 0.1, 1.0]
    frequency_list = [1, 2, 4, 8, 16, 32]
    format_string = sys.argv[2]
    duplicate_writing.duplicate_exact_lines(
            input_filename,
            proportion_list,
            frequency_list,
            format_string)
