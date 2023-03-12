#!/bin/bash

count=0

for file in pd_points/*.dat
do
    lc=`wc -l $file | cut -f1 -d' '`
    if [[ lc -eq 1024 ]]
    then
        if [[ count -lt 4096 ]]
        then
            cp $file pd_points_1024
        fi
        ((++count))
    fi
done
