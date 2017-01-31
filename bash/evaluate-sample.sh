#!/bin/sh
fileprefix=$1
let "fileidx = $2 % 50"
filelist=(../sample_template_input/$fileprefix*.seq)
seqfile=${filelist[fileidx]}
bname=`basename $seqfile .seq`
for k in 5 10 20 40 80; do
    outputprefix=../sample_template_output/$bname-$k
    if [ ! -f $outputprefix.trainprob ]; then
        ~/Mallet/bin/mallet evaluate-topics --input ../test_input/$fileprefix-test.seq --evaluator $outputprefix.evaluator --output-prob $outputprefix.testprob --output-doc-probs $outputprefix.testdocprobs
        ~/Mallet/bin/mallet evaluate-topics --input $seqfile --evaluator $outputprefix.evaluator --output-prob $outputprefix.trainprob --output-doc-probs $outputprefix.traindocprobs
    fi
done
