#!/bin/bash

if [[ $# -ne 3 ]]
then
    echo "Usage: $0 <source dir> <dest dir> <num to copy>"
    exit
fi

count=0

for file in "$1"/*
do
    if [[ count -lt $3 ]]
    then
        echo "Moving $file to $2"
        mv "$file" "$2"
    fi
    ((++count))
done
