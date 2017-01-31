#!/bin/sh
fileprefix=$1
let "fileidx = $2 % 300"
filelist=(../exact_duplicates_input/$fileprefix*.seq)
seqfile=${filelist[fileidx]}
bname=`basename $seqfile .seq`
for k in 5 10 20 40 80; do
    outputprefix=../exact_duplicates_output/$bname-$k
    if [ ! -f $outputprefix.doctopics ]; then
        ~/Mallet/bin/mallet train-topics --input $seqfile --num-topics $k --input-state $outputprefix.gz --no-inference --output-doc-topics $outputprefix.doctopics
    fi
done
