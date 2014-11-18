#pragma once

#include <mpi.h>

#include "common.h"
#include "halo-exchanger.h"

//see the vanilla version of this code for details about how this class operates
class ComputeInteractionsDPD : public HaloExchanger
{   
    //temporary buffer to compute accelerations in the halo
    Acceleration * acc_remote;

    //cuda-sync after to wait for packing of the halo
    void dpd_remote_interactions_stage1(const Particle * const p, const int n);

    //mpi-sync for the surrounding halos
    void dpd_remote_interactions_stage2(const Particle * const p, const int n, int saru_tag1, Acceleration * const a);

    cudaStream_t streams[7];
    int code2stream[26];
    
public:
    
    ComputeInteractionsDPD(MPI_Comm cartcomm, int L);
    
    ~ComputeInteractionsDPD();

    void evaluate(int& saru_tag, const Particle * const p, int n, Acceleration * const a,
		  const int * const cellsstart, const int * const cellscount);
};
