#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Xanda Schofield

"""mallet_reader.py

A class allowing reading of documents from a MALLET-formatted file.
"""
import re


class MalletReader():

    def __init__(self, filename):
        """Creates a new MalletReader for the given file.

        Arguments:
            filename::string -- the name of the file to read
        """
        self.filename = filename
        self.open_file = open(filename)
        self.line_regex = r'([^\t]+)\t([^\t]+)\t(.*)'

    def next(self):
        """Read a line from the reader, including only the text."""
        while True:
            line = self.open_file.readline()
            line_match = line.split('\t')
            if len(line_match) == 3:
                return line_match[2][:-1]

    def read_doc(self):
        """Alias for __next__."""
        return self.next()

    def __iter__(self):
        """Allows iteration support, e.g. for __ in __."""
        return self

    def read_doc_with_id(self):
        """Read a line from the reader as a tuple of (id, text)."""
        while True:
            line = self.open_file.readline()
            line_match = line.split('\t')
            if len(line_match) == 3:
                return line_match[2][:-1]

    def read(self):
        return '\n'.join([line for line in self])
