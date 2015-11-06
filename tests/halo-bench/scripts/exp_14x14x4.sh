#!/bin/bash -l
#
#SBATCH --job-name="exp_14x14x4"
#SBATCH --output=exp_14x14x4-%j.txt
#SBATCH --error=exp_14x14x4-%j.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=784
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=4

# !/usr/local/bin/bash

for k in 1 2 
do

echo "ISEND"
for i in 1 8 128 1024 2048 4096 8192 16384 32768 65536 131072
do
	echo $i
	aprun -n 784 -N 1 ${PWD}/halo_bench_isend 14 14 4 $i
done

done

#echo "SEND"
#for i in 1 8 128 1024 16384 65536 131072
#do
#	echo $i
#	aprun -n 784 -N 1 ${PWD}/halo_bench_send 14 14 4 $i
#done

#echo "PERSIST"
#for i in 1 8 128 1024 16384 65536 131072
#do
#	echo $i
#	aprun -n 784 -N 1 ${PWD}/halo_bench_persist 14 14 4 $i
#done
