#!/bin/bash -l
#
#SBATCH --job-name="exp_12x12x2"
#SBATCH --output=exp_12x12x2-%j.txt
#SBATCH --error=exp_12x12x2-%j.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=288
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=4

# !/usr/local/bin/bash

echo "ISEND"
for i in 1 8 128 1024 16384 65536 131072
do
	echo $i
	aprun -n 288 -N 1 ${PWD}/halo_bench_isend 12 12 2 $i
done

echo "SEND"
for i in 1 8 128 1024 16384 65536 131072
do
	echo $i
	aprun -n 288 -N 1 ${PWD}/halo_bench_send 12 12 2 $i
done

echo "PERSIST"
for i in 1 8 128 1024 16384 65536 131072
do
	echo $i
	aprun -n 288 -N 1 ${PWD}/halo_bench_persist 12 12 2 $i
done
