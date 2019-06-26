#!/bin/bash

ncdir=$1
prefix=$2
zdir=$ncdir'/zipped_files/'$prefix'*'

for f in $zdir
do
    echo $f
    tar --skip-old-files -xzvf $f --strip=1 -C $ncdir
done
