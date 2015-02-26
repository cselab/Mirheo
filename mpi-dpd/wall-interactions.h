#pragma once

#include <mpi.h>

#include <../dpd-rng.h>

#include "common.h"

namespace SolidWallsKernel
{
    __global__ void fill_keys(const Particle * const particles, const int n, const int L, int * const key);
}

class ComputeInteractionsWall
{
    MPI_Comm cartcomm;
    int L, myrank, dims[3], periods[3], coords[3];
 
    Logistic::KISS trunk;
   
    int solid_size;
    Particle * solid;

    cudaArray * arrSDF;

    CellLists cells;
    
public:

    ComputeInteractionsWall(MPI_Comm cartcomm, const int L, Particle* const p, const int n, int& nsurvived);

    ~ComputeInteractionsWall();
     
    void bounce(Particle * const p, const int n);

    void interactions(const Particle * const p, const int n, Acceleration * const acc,
		      const int * const cellsstart, const int * const cellscount);
};
