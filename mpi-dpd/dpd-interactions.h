#pragma once

#include <utility>
#include <mpi.h>

#include <map>
#include <string>

#include <../dpd-rng.h>

#include "common.h"
#include "halo-exchanger.h"

//see the vanilla version of this code for details about how this class operates
class ComputeInteractionsDPD : public HaloExchanger
{           
    HookedTexture texSC[26], texDC[26], texSP[26];
    
    //temporary buffer to compute accelerations in the halo
    SimpleDeviceBuffer<Acceleration> acc_remote[26];

    Logistic::KISS local_trunk;
    Logistic::KISS interrank_trunks[26];
    bool interrank_masks[26];

    cudaEvent_t evremoteint[7];

    cudaStream_t streams[7];
    int code2stream[26];
    
public:
    
    ComputeInteractionsDPD(MPI_Comm cartcomm);

    void remote_interactions(const Particle * const p, const int n, Acceleration * const a, cudaStream_t stream);

    void local_interactions(const Particle * const p, const int n, Acceleration * const a,
			   const int * const cellsstart, const int * const cellscount, cudaStream_t stream);

    ~ComputeInteractionsDPD();
};
