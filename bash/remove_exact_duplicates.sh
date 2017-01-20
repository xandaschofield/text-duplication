#!/bin/sh
# Removes exact duplicate lines in file $ARG1 and shuffles the remaining
# lines before writing them to file $ARG2
cat $1 | sort | uniq | shuf > $2
