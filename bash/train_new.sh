#!/bin/sh
# Arguments: directory to work in, prefix, index, total valid files count
filepath=$1
fileprefix=$2
let "fileidx = $3 % $4"
filelist=($filepath/input/$fileprefix*.seq)
seqfile=${filelist[fileidx]}
bname=`basename $seqfile .seq`
for k in 5 10 20 40 80 160 320; do
    outputprefix=$filepath/output/$bname-$k
    if [ ! -f $outputprefix.gz ]; then
        echo train $outputprefix $seqfile
        mallet train-topics --input $seqfile --num-topics $k --output-state $outputprefix.gz --output-topic-keys $outputprefix.keys --num-top-words 20 --diagnostics-file $outputprefix.diag.xml --evaluator-filename $outputprefix.evaluator --word-topic-counts-file $outputprefix.wordcounts.txt --topic-word-weights-file $outputprefix.wordweights.txt --output-doc-topics $outputprefix.doctopics &> $outputprefix.out
    elif [[ ! -f $outputprefix.keys || ! -f $outputprefix.diag.xml || ! -f $outputprefix.evaluator || ! -f $outputprefix.wordcounts.txt || ! -f $outputprefix.wordweights.txt || ! -f $outputprefix.doctopics ]]; then
        echo repair $outputprefix $seqfile
        mallet train-topics --input $seqfile --num-topics $k --input-state $outputprefix.gz --no-inference --output-topic-keys $outputprefix.keys --num-top-words 20 --diagnostics-file $outputprefix.diag.xml --evaluator-filename $outputprefix.evaluator --word-topic-counts-file $outputprefix.wordcounts.txt --topic-word-weights-file $outputprefix.wordweights.txt --output-doc-topics $outputprefix.doctopics
    fi
done
