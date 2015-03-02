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
    class HookedTexture
    {
	std::pair< void *, int> registered;
    
	template<typename T>  void _create(T * data, const int n)
	{
	    struct cudaResourceDesc resDesc;
	    memset(&resDesc, 0, sizeof(resDesc));
	    resDesc.resType = cudaResourceTypeLinear;
	    resDesc.res.linear.devPtr = data;
	    resDesc.res.linear.sizeInBytes = n * sizeof(T);
	    resDesc.res.linear.desc = cudaCreateChannelDesc<T>();
		
	    struct cudaTextureDesc texDesc;
	    memset(&texDesc, 0, sizeof(texDesc));
	    texDesc.addressMode[0]   = cudaAddressModeWrap;
	    texDesc.addressMode[1]   = cudaAddressModeWrap;
	    texDesc.filterMode       = cudaFilterModePoint;
	    texDesc.readMode         = cudaReadModeElementType;
	    texDesc.normalizedCoords = 0;
		
	    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
	}
	
	void _discard()	{  if (texObj != 0)CUDA_CHECK(cudaDestroyTextureObject(texObj)); }
	    
    public:
	
	cudaTextureObject_t texObj;
	
    HookedTexture(): texObj(0) { }

	template<typename T>
	    cudaTextureObject_t acquire(T * data, const int n)
	{
	    std::pair< void *, int> target = std::make_pair(data, n);

	    if (target != registered)
	    {
		_discard();
		_create(data, n);
		registered = target;
	    }

	    return texObj;
	}
	
	~HookedTexture() { _discard(); }
    };

    struct LocalWorkParams
    {
	float seed1;
	const Particle * p;
	int n;
	Acceleration *  a;
	const int *  cellsstart;
	const int *  cellscount;

    LocalWorkParams(): seed1(-1), p(NULL), n(0), a(NULL), cellsstart(NULL), cellscount(NULL) {}

    LocalWorkParams(const float seed1, const Particle * const p, const int n, Acceleration * const a,
		    const int * const cellsstart, const int * const cellscount):
	seed1(seed1), p(p), n(n), a(a), cellsstart(cellsstart), cellscount(cellscount) { }
	
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
    
public:
    
    ComputeInteractionsDPD(MPI_Comm cartcomm, int L);

    void evaluate(const Particle * const p, int n, Acceleration * const a,
		  const int * const cellsstart, const int * const cellscount);
};
