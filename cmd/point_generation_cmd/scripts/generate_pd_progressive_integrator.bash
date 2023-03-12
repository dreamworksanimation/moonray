#!/bin/bash


outdir='ppd_points'
sets=4096
setsize=1024
seedfile=poisson_seed_4096.dat

mkdir -p "$outdir"

counter=1
while [[ $counter -le $sets ]]
do
    seedpoint=$(sed -n "${counter}p" "$seedfile")
    filename=$(printf "$outdir/points%04u.dat" "$counter")
    ./pd_progressive_generation -count "$setsize" -out "$filename" -point $seedpoint
    ((++counter))
done

