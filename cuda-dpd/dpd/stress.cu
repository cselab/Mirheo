/*
 *  stress.cu
 *  Part of uDeviceX/cuda-dpd-sem/dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-09-29.
 *  Copyright 2015. All rights reserved.
 *
 *  Users are NOT authorized
 *  to employ the present software for their own publications
 *  before getting a written permission from the author of this file.
 */

#include <cstdio>
#include <cassert>

#include "cuda-dpd.h"
#include "../dpd-rng.h"
#include "../hacks.h"

namespace StressKernels
{
    struct InfoStress
    {
	int3 ncells;
	float aij, gamma, sigmaf;
	float *sigma_xx, *sigma_xy, *sigma_xz, *sigma_yy, *sigma_yz, *sigma_zz, *axayaz;
	float seed;
    };

    __constant__ InfoStress info; 
    
    texture<float2, cudaTextureType1D> texParticles;
    texture<int, cudaTextureType1D> texStart, texCount;

#define _XCPB_ 2
#define _YCPB_ 2
#define _ZCPB_ 1
#define CPB (_XCPB_ * _YCPB_ * _ZCPB_)

    __device__ float3 _dpd_interaction(const int dpid, const float3 xdest, const float3 udest, const int spid, const float2 stmp0, const float2 stmp1)
    {
	const int sentry = 3 * spid;
	const float2 stmp2 = tex1Dfetch(texParticles, sentry + 2);

	const float _xr = xdest.x - stmp0.x;
	const float _yr = xdest.y - stmp0.y;
	const float _zr = xdest.z - stmp1.x;
	const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	assert(rij2 < 1);

	const float invrij = rsqrtf(rij2);
	const float rij = rij2 * invrij;
	const float argwr = 1 - rij;
	const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);

	const float xr = _xr * invrij;
	const float yr = _yr * invrij;
	const float zr = _zr * invrij;

	const float rdotv =
	    xr * (udest.x - stmp1.y) +
	    yr * (udest.y - stmp2.x) +
	    zr * (udest.z - stmp2.y);

	const float myrandnr = Logistic::mean0var1(info.seed, min(spid, dpid), max(spid, dpid));

	const float strength = info.aij * argwr - (info.gamma * wr * rdotv + info.sigmaf * myrandnr) * wr;

	return make_float3(strength * xr, strength * yr, strength * zr);
    }

#define __IMOD(x,y) ((x)-((x)/(y))*(y))

    template<int COLS, int ROWS>
    __global__ void stress_kernel()
    {
	int mycount = 0, myscan = 0;

	__shared__ int volatile starts[CPB][16], scan[CPB][16];

	if (threadIdx.x < 14)
	{
	    const int cbase = blockIdx.x * blockDim.y + threadIdx.y;

	    int dx, dy, dz;
	    dx = dy = dz = threadIdx.x / 3;
	    dx = threadIdx.x - dx * 3 - 1;
	    dy = __IMOD(dy, 3) - 1;
	    dz = __IMOD(dz / 3, 3) - 1;

	    int cid = cbase + dz * info.ncells.x * info.ncells.y + dy * info.ncells.x + dx;

	    const bool valid_cid = (cid >= 0) && (cid < info.ncells.x * info.ncells.y * info.ncells.z);

	    starts[threadIdx.y][threadIdx.x] = (valid_cid) ? tex1Dfetch(texStart, cid) : 0;

	    myscan = mycount = (valid_cid) ? tex1Dfetch(texCount, cid) : 0;
	}

#pragma unroll
	for(int L = 1; L < 16; L <<= 1)
	    myscan += (threadIdx.x >= L) * __shfl_up(myscan, L);

	if (threadIdx.x < 15)
	    scan[threadIdx.y][threadIdx.x] = myscan - mycount;

	const int subtid = threadIdx.x % COLS;
	const int slot = threadIdx.x / COLS;

	const int dststart = starts[threadIdx.y][13];
	const int lastdst = dststart + scan[threadIdx.y][14] - scan[threadIdx.y][13];

	const int nsrc = scan[threadIdx.y][14];
	const int nsrcext = scan[threadIdx.y][13];

	for(int pid = subtid; pid < nsrc; pid += COLS)
	{
	    const int key9 = 9 * (pid >= scan[threadIdx.y][9]);

	    int key3 = 3 * (pid >= scan[threadIdx.y][key9 + 3]);
	    key3 += (key9 < 9) ? 3 * (pid >= scan[threadIdx.y][key9 + 6]) : 0;

	    int spid = pid - scan[threadIdx.y][key3 + key9] + starts[threadIdx.y][key3 + key9];

	    const int sentry = 3 * spid;
	    const float2 stmp0 = tex1Dfetch(texParticles, sentry);
	    const float2 stmp1 = tex1Dfetch(texParticles, sentry + 1);

	    for(int dpid = dststart + slot; dpid < lastdst; dpid += ROWS)
	    {
		float3 xdest, udest;

		float2 dtmp0 = tex1Dfetch(texParticles, 3 * dpid);
		xdest.x = dtmp0.x;
		xdest.y = dtmp0.y;

		dtmp0 = tex1Dfetch(texParticles, 3 * dpid + 1);
		xdest.z = dtmp0.x;
		udest.x = dtmp0.y;

		dtmp0 = tex1Dfetch(texParticles, 3 * dpid + 2);
		udest.y = dtmp0.x;
		udest.z = dtmp0.y;

		const float rx = xdest.x - stmp0.x;
		const float ry = xdest.y - stmp0.y;
		const float rz = xdest.z - stmp1.x;
		
		const float d2 = rx * rx + ry * ry + rz * rz;

		if ((dpid != spid) && (d2 < 1.0f))
		{
		    const float3 f = _dpd_interaction(dpid, xdest, udest, spid, stmp0, stmp1);

		    atomicAdd(info.sigma_xx + dpid, f.x * rx);
		    atomicAdd(info.sigma_xy + dpid, f.x * ry);
		    atomicAdd(info.sigma_xz + dpid, f.x * rz);
		    atomicAdd(info.sigma_yy + dpid, f.y * ry);
		    atomicAdd(info.sigma_yz + dpid, f.y * rz);
		    atomicAdd(info.sigma_zz + dpid, f.z * rz);
		    
		    if (info.axayaz)
		    {
			atomicAdd(info.axayaz + 3 * dpid    , f.x);
			atomicAdd(info.axayaz + 3 * dpid + 1, f.y);
			atomicAdd(info.axayaz + 3 * dpid + 2, f.z);
		    }
		    
		    if (pid < nsrcext)
		    {
			atomicAdd(info.sigma_xx + spid, f.x * rx);
			atomicAdd(info.sigma_xy + spid, f.x * ry);
			atomicAdd(info.sigma_xz + spid, f.x * rz);
			atomicAdd(info.sigma_yy + spid, f.y * ry);
			atomicAdd(info.sigma_yz + spid, f.y * rz);
			atomicAdd(info.sigma_zz + spid, f.z * rz);
			  
			if (info.axayaz)
			{
			    atomicAdd(info.axayaz + 3*spid    , -f.x);
			    atomicAdd(info.axayaz + 3*spid + 1, -f.y);
			    atomicAdd(info.axayaz + 3*spid + 2, -f.z);
			}
		    }
		}
	    }
	}
    }
    
    bool computestress_init = false;
}

using namespace StressKernels;

void compute_stress(const float * const xyzuvw,
		    const int np,
		    const int * const cellsstart, const int * const cellscount,
		    const int XL, const int YL, const int ZL,
		    const float aij, const float gamma,  const float sigmaf, const float seed,
		    float * const sigma_xx, float * const sigma_xy, float * const sigma_xz,
		    float * const sigma_yy, float * const sigma_yz, float * const sigma_zz,
		    float * const axayaz, 
    		    cudaStream_t stream)
{
    if (np == 0)
    {
	printf("WARNING: stress_nohost called with np = %d\n", np);
	return;
    }
	
    if (!computestress_init)
    {
	texStart.channelDesc = cudaCreateChannelDesc<int>();
	texStart.filterMode = cudaFilterModePoint;
	texStart.mipmapFilterMode = cudaFilterModePoint;
	texStart.normalized = 0;

	texCount.channelDesc = cudaCreateChannelDesc<int>();
	texCount.filterMode = cudaFilterModePoint;
	texCount.mipmapFilterMode = cudaFilterModePoint;
	texCount.normalized = 0;

	texParticles.channelDesc = cudaCreateChannelDesc<float2>();
	texParticles.filterMode = cudaFilterModePoint;
	texParticles.mipmapFilterMode = cudaFilterModePoint;
	texParticles.normalized = 0;

	CUDA_CHECK(cudaFuncSetCacheConfig(stress_kernel<32, 1>, cudaFuncCachePreferL1));
        
	computestress_init = true;
    }

    size_t textureoffset;
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texParticles, xyzuvw, &texParticles.channelDesc, sizeof(float) * 6 * np));
    assert(textureoffset == 0);

    const int ncells = XL * YL * ZL;
    
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texStart, cellsstart, &texStart.channelDesc, sizeof(int) * ncells));
    assert(textureoffset == 0);
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texCount, cellscount, &texCount.channelDesc, sizeof(int) * ncells));
    assert(textureoffset == 0);

    {
	static InfoStress c = { make_int3(XL, YL, ZL), aij, gamma, sigmaf,
				sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz,
				axayaz, seed };

	CUDA_CHECK(cudaMemcpyToSymbolAsync(info, &c, sizeof(c), 0, cudaMemcpyHostToDevice, stream));
    }
    
    if (axayaz)
	CUDA_CHECK(cudaMemsetAsync(axayaz, 0, sizeof(float) * 3 * np, stream));

    float * const ptrs[] = { sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz };

    for(int c = 0; c < 6; ++c)
	CUDA_CHECK(cudaMemsetAsync(ptrs[c], 0, sizeof(float) * np, stream));
    
    stress_kernel<32, 1><<<(ncells + CPB - 1) / CPB, dim3(32, CPB), 0, stream>>>();

    CUDA_CHECK(cudaPeekAtLastError());
}

