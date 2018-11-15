#! /bin/sh

DATA=results_daint/
FIG=../docs/source/images/
EXT=png

./tools/plot_weak.py --files $DATA/poiseuille/weak_*.txt --ref 1 --out $FIG/weak_solvent.$EXT
./tools/plot_weak.py --files $DATA/blood/weak_*.txt      --ref 1 --out $FIG/weak_blood.$EXT
./tools/plot_weak.py --files $DATA/walls/weak_*.txt      --ref 1 --out $FIG/weak_walls.$EXT
./tools/plot_weak.py --files $DATA/rigids/weak_*.txt     --ref 1 --out $FIG/weak_rigids.$EXT

./tools/plot_strong.py --files $DATA/poiseuille/strong_*.txt --out $FIG/strong_solvent.$EXT
./tools/plot_strong.py --files $DATA/blood/strong_*.txt      --out $FIG/strong_blood.$EXT
./tools/plot_strong.py --files $DATA/walls/strong_*.txt      --out $FIG/strong_walls.$EXT
