#pragma once

#include <utility>
#include <mpi.h>

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
   
    HookedTexture texSC[26], texDC[26], texSP[26];
    
    //temporary buffer to compute accelerations in the halo
    SimpleDeviceBuffer<Acceleration> acc_remote[26];

    //mpi-sync for the surrounding halos
    void dpd_remote_interactions(const Particle * const p, const int n, int saru_tag1, Acceleration * const a);
    
public:
    
    ComputeInteractionsDPD(MPI_Comm cartcomm, int L);

    void evaluate(int& saru_tag, const Particle * const p, int n, Acceleration * const a,
		  const int * const cellsstart, const int * const cellscount);
};
