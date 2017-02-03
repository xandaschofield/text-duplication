#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield
import sys

import duplicate_writing


if __name__ == "__main__":
    input_filename = sys.argv[1]
    proportion_list = [None]
    frequency_list = [2**x for x in range(15)]
    format_string = sys.argv[2]
    duplicate_writing.duplicate_exact_lines(
            input_filename,
            proportion_list,
            frequency_list,
            format_string)
