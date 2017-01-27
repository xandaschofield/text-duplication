#!/bin/sh
for filename in $1*.txt; do
    pathlength=${#filename}
    basepath=${filename:0:(pathlength-4)}
    mallet import-file --input $filename --output $basepath.seq --keep-sequence
done
