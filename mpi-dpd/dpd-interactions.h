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
  

    struct LocalWorkParams
    {
	float seed1;
	const Particle * p;
	int n;
	Acceleration *  a;
	const int *  cellsstart;
	const int *  cellscount;
	cudaStream_t stream;

    LocalWorkParams(): seed1(-1), p(NULL), n(0), a(NULL), cellsstart(NULL), cellscount(NULL), stream(0) {}

    LocalWorkParams(const float seed1, const Particle * const p, const int n, Acceleration * const a,
		    const int * const cellsstart, const int * const cellscount, cudaStream_t stream):
	seed1(seed1), p(p), n(n), a(a), cellsstart(cellsstart), cellscount(cellscount), stream(stream) { }
	
    } localwork;
            
    HookedTexture texSC[26], texDC[26], texSP[26];
    
    //temporary buffer to compute accelerations in the halo
    SimpleDeviceBuffer<Acceleration> acc_remote[26];

    Logistic::KISS local_trunk;
    Logistic::KISS interrank_trunks[26];
    bool interrank_masks[26];

    //mpi-sync for the surrounding halos
    void dpd_remote_interactions(const Particle * const p, const int n, Acceleration * const a);

    void spawn_local_work();

    cudaEvent_t evmerge;
    
public:
    
    ComputeInteractionsDPD(MPI_Comm cartcomm);

    void evaluate(const Particle * const p, int n, Acceleration * const a,
		  const int * const cellsstart, const int * const cellscount, cudaStream_t stream);

    ~ComputeInteractionsDPD();
};
