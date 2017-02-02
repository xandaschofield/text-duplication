#!/bin/sh
fileprefix=$1
let "fileidx = $2 % 50"
filelist=(../lorem_ipsum_input/$fileprefix*.txt)
txtfile=${filelist[fileidx]}
bname=`basename $txtfile .txt`
for k in 5 10 20 40 80; do
    outputprefix=../lorem_ipsum_output/$bname-$k
    python ../python/lsa.py $txtfile $k $outputprefix
done
