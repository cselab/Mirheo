#! /usr/local/bin/bash

unitXRes=32
unitYRes=128
unitZRes=64

nColumns=2
# nRows is defined by files.txt

for i in `seq 3 15`; do
    ../sdf-unit-par/sdf-unit $unitXRes $unitYRes 24 96 $i gap$i.dat
done

for i in `seq 3 15`; do
    ../sdf-collage/sdf-collage gap$i.dat $nColumns 1 r$i.dat
done

../sdf-collage/sdf-collage files.txt 1 1 collage.dat
../2Dto3D/2Dto3D collage.dat 40.0 4.0 $unitZRes sdf.dat
