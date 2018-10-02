#! /bin/sh

launch() (
    NX=$1;     shift
    LX=$1;     shift
    script=$1; shift

    NY=$NX; NZ=$NX

    exName="main.py"
    
    baseDir=$SCRATCH/benchmarks/weak/bulk_solvent/
    name=${N}_${LX}
    runDir=$baseDir/$name
    
    mkdir -p $runDir
    cp $script $runDir/$exName
    cd $runDir    

    tot=#TODO
    
    . udx.load

    grid_order -R -g $NX,$NY,$NZ -H | perl -p -e 's/(\d+)/(2*$1).",".(2*$1+1)/eg unless (/\#/);' > MPICH_RANK_ORDER 
    
    sbatch <<!!!
#!/bin/bash -l
#
#SBATCH --job-name="weak_$name"
#SBATCH --time=00:20:00
#SBATCH --nodes=$tot
#SBATCH --constraint=gpu
#SBATCH --account=ch7
#SBATCH --core-spec=4
#SBATCH --contiguous
#SBATCH --output=out.txt
#SBATCH --error=err.txt
    
export MPICH_MAX_THREAD_SAFETY=multiple
export MPICH_NEMESIS_ASYNC_PROGRESS=1

if [ $tot -gt 1 ]; then
    export MPICH_RANK_REORDER_DISPLAY=1
    export MPICH_RANK_REORDER_METHOD=3
fi

srun -u --ntasks-per-node 2 ./$exName
!!!
)

