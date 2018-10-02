#!/bin/sh

baseDir=$1; shift

for d in `ls $baseDir`; do
    f=$baseDir/$d/stats.txt
    if test -s $f; then
        time=`./tools/averageTime.py --file $f`
	echo $d $time
    fi
done
