#! /bin/sh

DATA=results_daint/
FIG=../docs/source/images/
EXT=png

./tools/plot_weak.py --files $DATA/fine_min_weak_periodic_poiseuille_* --ref 1 --out $FIG/weak_solvent.$EXT
./tools/plot_weak.py --files $DATA/fine_min_weak_periodic_blood_* --ref 1 --out $FIG/weak_blood.$EXT

./tools/plot_strong.py --files $DATA/strong_periodic_poiseuille_* --out $FIG/strong_solvent.$EXT
./tools/plot_strong.py --files $DATA/strong_periodic_blood_* --out $FIG/strong_blood.$EXT
