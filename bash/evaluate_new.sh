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
    if [ ! -f $outputprefix.traindocprobs ]; then
        echo eval $outputprefix $seqfile
        ~/Mallet/bin/mallet evaluate-topics --input $seqfile --evaluator $outputprefix.evaluator --output-prob $outputprefix.trainprob --output-doc-probs $outputprefix.traindocprobs
    fi
done
