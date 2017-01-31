#!/bin/sh
fileprefix=$1
let "fileidx = $2 % 300"
filelist=(../exact_duplicates_input/$fileprefix*.txt)
txtfile=${filelist[fileidx]}
bname=`basename $txtfile .txt`
for k in 5 10 20 40 80; do
    outputprefix=../exact_duplicates_output/$bname-$k
    python ../python/lsa.py $txtfile $k $outputprefix
done
