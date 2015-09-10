/*
 *  hacks.h
 *  Part of uDeviceX/cuda-dpd-sem/
 *
 *  Created and authored by Diego Rossinelli on 2014-07-29.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#pragma once
#include <unistd.h>
#include <cstdio>

#ifndef CUDA_CHECK
#define CUDA_CHECK(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
	fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	sleep(5);
	if (abort) exit(code);
    }
}
#endif

#include <thrust/device_vector.h>
template<typename T> T * _ptr(thrust::device_vector<T>& v)
{
    return thrust::raw_pointer_cast(v.data());
}

struct TextureWrap
{
    cudaTextureObject_t texObj;

    template<typename ElementType>
    TextureWrap(ElementType * data, const int n):
	texObj(0)
	{
	    struct cudaResourceDesc resDesc;
	    memset(&resDesc, 0, sizeof(resDesc));
	    resDesc.resType = cudaResourceTypeLinear;
	    resDesc.res.linear.devPtr = data;
	    resDesc.res.linear.sizeInBytes = n * sizeof(ElementType);
	    resDesc.res.linear.desc = cudaCreateChannelDesc<ElementType>();
    
	    struct cudaTextureDesc texDesc;
	    memset(&texDesc, 0, sizeof(texDesc));
	    texDesc.addressMode[0]   = cudaAddressModeWrap;
	    texDesc.addressMode[1]   = cudaAddressModeWrap;
	    texDesc.filterMode       = cudaFilterModePoint;
	    texDesc.readMode         = cudaReadModeElementType;
	    texDesc.normalizedCoords = 0;

	    texObj = 0;
	    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
	}

    ~TextureWrap()
	{
	    CUDA_CHECK(cudaDestroyTextureObject(texObj));
	}
};

static int saru_tid = 0;
