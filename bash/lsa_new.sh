#!/bin/sh
filepath=$1
fileprefix=$2
let "fileidx = $3 % $4"
outpath=$5 # TODO: switch back to filepath
filelist=($filepath/input/$fileprefix*.txt)
txtfile=${filelist[fileidx]}
bname=`basename $txtfile .txt`
for k in 5 10 20 40 80 160 320; do
    outputprefix=$outpath/output/$bname-$k
    python ../python/lsa.py $txtfile $k $outputprefix
done
