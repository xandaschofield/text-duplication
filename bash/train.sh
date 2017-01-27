#!/bin/sh
fileprefix=$1
fileidx=$2
filelist=(../mallet_inputs/$fileprefix*.seq)
seqfile=${filelist[fileidx]}
bname=`basename $seqfile .seq`
for k in 5 10 20 40 80; do
    outputprefix=../opt_exact_output/$bname-$k
    ~/Mallet/bin/mallet train-topics --input $seqfile --num-topics $k --output-state $outputprefix.gz --output-topic-keys $outputprefix.keys --num-top-words 20 --diagnostics-file $outputprefix.diag.xml --evaluator-filename $outputprefix.evaluator --word-topic-counts-file $outputprefix.wordcounts.txt --topic-word-weights-file $outputprefix.wordweights.txt &> $outputprefix.out
done
