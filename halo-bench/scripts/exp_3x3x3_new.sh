#!/bin/bash -l
#
#SBATCH --job-name="exp_3x3x3"
#SBATCH --output=exp_3x3x3-%j.txt
#SBATCH --error=exp_3x3x3-%j.txt
#SBATCH --time=01:00:00
#SBATCH --nodes=27
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=8

# !/usr/local/bin/bash

export MPICH_MAX_THREAD_SAFETY=multiple
export MPICH_NEMESIS_ASYNC_PROGRESS=1

for k in 1 2
do

echo "ISEND"
for i in 0 # 100 200 300 400 
do
	echo $i
	aprun -n 27 -N 1 ${PWD}/halo_bench_isend 3 3 3 0 $i 2000 0.0
done

done

#echo "SEND"
#for i in 1 8 128 1024 16384 65536 131072
#do
#	echo $i
#	aprun -n 27 -N 1 ${PWD}/halo_bench_send 3 3 3 $i
#done

#echo "PERSIST"
#for i in 1 8 128 1024 16384 65536 131072
#do
#	echo $i
#	aprun -n 27 -N 1 ${PWD}/halo_bench_persist 3 3 3 $i
#done
