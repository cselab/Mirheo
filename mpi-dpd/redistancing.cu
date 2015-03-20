/*
 *  redistancing.cu
 *  Part of CTC/mpi-dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-03-17.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include "redistancing.h"

#include "common.h"

#define _ACCESS(ary, x, y, z) tex3D(tex##ary, x, y, z)

namespace Redistancing
{
    texture<float, 3, cudaReadModeElementType> texphi0, texphi;

    struct Info
    {
	int NX, NY, NZ;
	float dx, dy, dz, invdx, invdy, invdz, cfl, dt, dls[27];
    };
    
    __constant__ Info info;

    template<int d>
    __device__ inline bool anycrossing_dir(int ix, int iy, int iz, const float sgn0) 
    {
        const int dx = d == 0, dy = d == 1, dz = d == 2;
	
        const float fm1 = _ACCESS(phi0, ix - dx, iy - dy, iz - dz);
        const float fp1 = _ACCESS(phi0, ix + dx, iy + dy, iz + dz);
	
        return (fm1 * sgn0 < 0 || fp1 * sgn0 < 0);
    }
    
    __device__  inline bool anycrossing(int ix, int iy, int iz, const float sgn0)
    {
        return
	    anycrossing_dir<0>(ix, iy, iz, sgn0) ||
	    anycrossing_dir<1>(ix, iy, iz, sgn0) ||
	    anycrossing_dir<2>(ix, iy, iz, sgn0) ;
    }

    __device__  inline float simple_scheme(int ix, int iy, int iz, float sgn0, float myphi0) 
    {
	float mindistance = 1e6f;
	
	for(int code = 0; code < 3 * 3 * 3; ++code)
	{
	    if (code == 1 + 3 + 9) continue;
	  
	    const int xneighbor = ix + (code % 3) - 1;
	    const int yneighbor = iy + (code % 9) / 3 - 1;
	    const int zneighbor = iz + (code / 9) - 1;
	    
	    if (xneighbor < 0 || xneighbor >= info.NX) continue;
	    if (yneighbor < 0 || yneighbor >= info.NY) continue;
	    if (zneighbor < 0 || zneighbor >= info.NZ) continue;
	    
	    const float phi0_neighbor = _ACCESS(phi0, xneighbor, yneighbor, zneighbor);
	    const float phi_neighbor = _ACCESS(phi, xneighbor, yneighbor, zneighbor);
	    
	    const float dl = info.dls[code];
	    
	    float distance = 0;
	    
	    if (sgn0 * phi0_neighbor < 0)
		distance = - myphi0 * dl / (phi0_neighbor - myphi0);
	    else
		distance = dl + abs(phi_neighbor);
	    
	    mindistance = min(mindistance, distance);
	}
	
	return sgn0 * mindistance;
    }

    __device__ float sussman_scheme(int ix, int iy, int iz, float sgn0) 
    {
        const float phicenter =  _ACCESS(phi, ix, iy, iz);
        
        const float dphidxm = phicenter -     _ACCESS(phi, ix - 1, iy, iz);
        const float dphidxp = _ACCESS(phi, ix + 1, iy, iz) - phicenter;
        const float dphidym = phicenter -     _ACCESS(phi, ix, iy - 1, iz);
        const float dphidyp = _ACCESS(phi, ix, iy + 1, iz) - phicenter;
        const float dphidzm = phicenter -     _ACCESS(phi, ix, iy, iz - 1);
        const float dphidzp = _ACCESS(phi, ix, iy, iz + 1) - phicenter;
 
	if (sgn0 == 1)
        {
	    const float xgrad0 = max( max((float)0, dphidxm), -min((float)0, dphidxp)) * info.invdx;
	    const float ygrad0 = max( max((float)0, dphidym), -min((float)0, dphidyp)) * info.invdy;
	    const float zgrad0 = max( max((float)0, dphidzm), -min((float)0, dphidzp)) * info.invdz;
	    
	    const float G0 = sqrtf(xgrad0 * xgrad0 + ygrad0 * ygrad0 + zgrad0 * zgrad0) - 1;
	    
	    return phicenter - info.dt * sgn0 * G0;
        }
        else
        {
	    const float xgrad1 = max( -min((float)0, dphidxm), max((float)0, dphidxp)) * info.invdx;
	    const float ygrad1 = max( -min((float)0, dphidym), max((float)0, dphidyp)) * info.invdy;
	    const float zgrad1 = max( -min((float)0, dphidzm), max((float)0, dphidzp)) * info.invdz;
	    
	    const float G1 = sqrtf(xgrad1 * xgrad1 + ygrad1 * ygrad1 + zgrad1 * zgrad1) - 1;
	    
	    return phicenter - info.dt * sgn0 * G1;
        }
    }
    
    __global__ void step(float * dst)
    {
	assert(blockDim.x * gridDim.x >= info.NX);
	assert(blockDim.y * gridDim.y >= info.NY);
	assert(blockDim.z * gridDim.z >= info.NZ);

	const int ix = threadIdx.x + blockDim.x * blockIdx.x;
	const int iy = threadIdx.y + blockDim.y * blockIdx.y;
	const int iz = threadIdx.z + blockDim.z * blockIdx.z;

	if (ix >= info.NX || iy >= info.NY || iz > info.NZ)
	    return;

	const float myval0 = _ACCESS(phi0, ix, iy, iz);
	
	float sgn0 = 0;

	if (myval0 > 0)
	    sgn0 = 1;
	else if (myval0 < 0)
	    sgn0 = -1;
	
	const bool boundary =  (
	    ix == 0 || ix == info.NX - 1 ||
	    iy == 0 || iy == info.NY - 1 ||
	    iz == 0 || iz == info.NZ - 1 );
	
	float val = _ACCESS(phi, ix, iy, iz); 

	if (boundary)
	    val = simple_scheme(ix, iy, iz, sgn0, myval0);
	else
	{
	    if( anycrossing(ix, iy, iz, sgn0) )
	    {
		//undisclosed code here, for now
		assert(!isnan(val));
		assert(!isinf(val));
	    }
	    else						
		val = sussman_scheme(ix, iy, iz, sgn0);
	}

	dst[ix + info.NX * (iy + info.NY * iz)] = val;
	
	assert(!isnan(val));
	assert(!isinf(val));
	assert(val * sgn0 >= 0); 
    }
}

void redistancing(float * host_inout, const int NX, const int NY, const int NZ, const float dx, const float dy, const float dz, 
		  const int niterations)
{
    if (niterations <= 0)
	return;
 
    {
	Redistancing::Info info;
	
	info.NX = NX;
	info.NY = NY;
	info.NZ = NZ;

	info.dx = dx;
	info.dy = dy;
	info.dz = dz;

	info.invdx = 1/dx;
	info.invdy = 1/dy;
	info.invdz = 1/dz;

	const float smallest_spacing = min(dx, min(dy, dz));
	info.cfl = 0.25f;
	info.dt = info.cfl * smallest_spacing;
		
	for(int code = 0; code < 3 * 3 * 3; ++code)
	{
	    if (code == 1 + 3 + 9) continue;
	    
	    const float deltax = dx * ((code % 3) - 1);
	    const float deltay = dy * ((code % 9) / 3 - 1);
	    const float deltaz = dz * ((code / 9) - 1);
	    
	    const float dl = sqrtf(deltax * deltax + deltay * deltay + deltaz * deltaz);
	    
	    info.dls[code] = dl;
	}

	CUDA_CHECK(cudaMemcpyToSymbol(Redistancing::info, &info, sizeof(info)));
    }

    Redistancing::texphi0.normalized = false;
    Redistancing::texphi0.filterMode = cudaFilterModePoint;
    Redistancing::texphi0.addressMode[0] = cudaAddressModeClamp;
    Redistancing::texphi0.addressMode[1] = cudaAddressModeClamp;
    Redistancing::texphi0.addressMode[2] = cudaAddressModeClamp;
    
    Redistancing::texphi.normalized = false;
    Redistancing::texphi.filterMode = cudaFilterModePoint;
    Redistancing::texphi.addressMode[0] = cudaAddressModeClamp;
    Redistancing::texphi.addressMode[1] = cudaAddressModeClamp;
    Redistancing::texphi.addressMode[2] = cudaAddressModeClamp;

    cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float>();

    cudaArray * arrPhi0;
    CUDA_CHECK(cudaMalloc3DArray (&arrPhi0, &fmt, make_cudaExtent(NX, NY, NZ)));

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr((void *)host_inout, NX * sizeof(float), NX, NY);
    copyParams.dstArray = arrPhi0;
    copyParams.extent   = make_cudaExtent(NX, NY, NZ);
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));
    CUDA_CHECK(cudaBindTextureToArray(Redistancing::texphi0, arrPhi0, fmt));

    cudaArray * arrPhi;
    CUDA_CHECK(cudaMalloc3DArray (&arrPhi, &fmt, make_cudaExtent(NX, NY, NZ)));
    copyParams.dstArray = arrPhi;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));
    CUDA_CHECK(cudaBindTextureToArray(Redistancing::texphi, arrPhi, fmt));

    SimpleDeviceBuffer<float> tmp(NX * NY * NZ);

    copyParams.srcPtr   = make_cudaPitchedPtr((void *)tmp.data, NX * sizeof(float), NX, NY);
    copyParams.kind     = cudaMemcpyDeviceToDevice;

    for(int t = 0; t <  niterations; ++t)
    {
	//printf("timestep %d\n", t);
	Redistancing::step<<< dim3( (NX + 7) / 8, (NY + 7) / 8, (NZ + 1) / 2), dim3(8, 8, 2) >>>(tmp.data);
	//CUDA_CHECK(cudaDeviceSynchronize());
	//CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaMemcpy3DAsync(&copyParams));
    }

    CUDA_CHECK(cudaPeekAtLastError());
    
    CUDA_CHECK(cudaMemcpy(host_inout, tmp.data, sizeof(float) * NX * NY * NZ, cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaUnbindTexture(Redistancing::texphi0));
    CUDA_CHECK(cudaUnbindTexture(Redistancing::texphi));
    CUDA_CHECK(cudaFreeArray(arrPhi0));
    CUDA_CHECK(cudaFreeArray(arrPhi));

    CUDA_CHECK(cudaPeekAtLastError());
}
