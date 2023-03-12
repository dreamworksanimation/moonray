#!/bin/bash

outdir='../points/pd_reordered'
mkdir -p "$outdir"

for file in ../points/pd_raw/*.dat
do
    newname=`basename $file`
    newname="$outdir"/"$newname"
    #../reorder -in "$file" -out "$newname" -mode disk -stochastic 0
    ../reorder -in "$file" -out "$newname" -mode basic -stochastic 32
done
