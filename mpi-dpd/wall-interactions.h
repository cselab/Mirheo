#pragma once

#include <mpi.h>

#include "common.h"

class ComputeInteractionsWall
{
    MPI_Comm cartcomm;
    int L, myrank, dims[3], periods[3], coords[3];
    
    int solid_size;
    Particle * solid;

    cudaArray * arrSDF;

    CellLists cells;
    
public:

    ComputeInteractionsWall(MPI_Comm cartcomm, const int L, Particle* const p, const int n, int& nsurvived);

    ~ComputeInteractionsWall();
     
    void bounce(Particle * const p, const int n);

    void interactions(const Particle * const p, const int n, Acceleration * const acc,
		      const int * const cellsstart, const int * const cellscount, int& saru_tag);
};
