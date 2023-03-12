#!/bin/bash

outdir='../points/pd_raw'
sets=4096
setsize=1024

mkdir -p "$outdir"

counter=1
seed=1
while [[ $counter -le $sets ]]
do
    tmpname='pd_temp.dat'
    output=$(../pd_generation -grid_size 55 -seed "$seed" -out "$tmpname" -quiet)

    # Bash string operator: delete everything to the left of the rightmost '>'
    # character.  The pd_generation program outputs a BUNCH of text, but it's
    # deceiving, since there are a million '\r's in there.
    output=${output##*>}
    numgen=$(echo "$output" | cut -f 1 -d '/')
    if [[ $numgen -eq $setsize ]]
    then
        filename=$(printf "$outdir/points%04u.dat" "$seed")
        mv "$tmpname" "$filename"
        echo "$counter"
        ((++counter))
    fi
    ((++seed))
done
