/*
 *  cuda-dpd.cu
 *  Part of CTC/cuda-dpd-sem/dpd/
 *
 *  Created and authored by Diego Rossinelli on 2015-02-26.
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

struct InfoDPD
{
    int3 ncells;
    float3 domainsize, invdomainsize, domainstart;
    float invrc, aij, gamma, sigmaf;
    float * axayaz;
    float seed;
};

__constant__ InfoDPD info;

texture<float2, cudaTextureType1D> texParticles2;
texture<int, cudaTextureType1D> texStart, texCount;
 
#define _XCPB_ 2
#define _YCPB_ 2
#define _ZCPB_ 1
#define CPB (_XCPB_ * _YCPB_ * _ZCPB_)
//#define  _TIME_PROFILE_
//#define _INSPECT_

// Mauro: set to 0
#if 1


__device__ float3 _dpd_interaction(const int dpid, const float3 xdest, const float3 udest, const int spid)
{
    const int sentry = 3 * spid;
    const float2 stmp0 = tex1Dfetch(texParticles2, sentry);
    const float2 stmp1 = tex1Dfetch(texParticles2, sentry + 1);
    const float2 stmp2 = tex1Dfetch(texParticles2, sentry + 2);
    
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

__device__ float3 _dpd_interaction(const int dpid, const float3 xdest, const float3 udest, const float3 xsrc, const float3 usrc, const int spid)
{
    const float _xr = xdest.x - xsrc.x;
    const float _yr = xdest.y - xsrc.y;
    const float _zr = xdest.z - xsrc.z;

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
	xr * (udest.x - usrc.x) +
	yr * (udest.y - usrc.y) +
	zr * (udest.z - usrc.z);

    const float myrandnr = Logistic::mean0var1(info.seed, min(spid, dpid), max(spid, dpid));

    const float strength = info.aij * argwr - (info.gamma * wr * rdotv + info.sigmaf * myrandnr) * wr;

    return make_float3(strength * xr, strength * yr, strength * zr);
}

template<int COLS, int ROWS, int NSRCMAX>
__device__ void core(const int nsrc, const int * const scan, const int * const starts, 
		     const int ndst, const int dststart)
{
   int srcids[NSRCMAX];
    for(int i = 0; i < NSRCMAX; ++i)
	srcids[i] = 0;
    
    int srccount = 0;   
    assert(ndst == ROWS);
    
    const int tid = threadIdx.x; 
    const int slot = tid / COLS;
    const int subtid = tid % COLS;

    const int dpid = dststart + slot;
    const int entry = 3 * dpid;
    const float2 dtmp0 = tex1Dfetch(texParticles2, entry);
    const float2 dtmp1 = tex1Dfetch(texParticles2, entry + 1);
    const float2 dtmp2 = tex1Dfetch(texParticles2, entry + 2);
    const float3 xdest = make_float3(dtmp0.x, dtmp0.y, dtmp1.x);
    const float3 udest = make_float3(dtmp1.y, dtmp2.x, dtmp2.y);
    
    float xforce = 0, yforce = 0, zforce = 0;

    for(int s = 0; s < nsrc; s += COLS)
    {
	const int pid = s + subtid;
	const int key9 = 9 * ((pid >= scan[9]) + (pid >= scan[18]));
	const int key3 = 3 * ((pid >= scan[key9 + 3]) + (pid >= scan[key9 + 6]));
	const int key = key9 + key3;	    
	
	const int spid = pid - scan[key] + starts[key];
	
	const int sentry = 3 * spid;
	const float2 stmp0 = tex1Dfetch(texParticles2, sentry);
	const float2 stmp1 = tex1Dfetch(texParticles2, sentry + 1);
	
	const float xdiff = xdest.x - stmp0.x;
	const float ydiff = xdest.y - stmp0.y;
	const float zdiff = xdest.z - stmp1.x;
	const bool interacting = (s + subtid < nsrc) && (dpid != spid) && (xdiff * xdiff + ydiff * ydiff + zdiff * zdiff < 1);
	
	srcids[srccount] = spid;	
	srccount += interacting;
	
	if (srccount == NSRCMAX)
	{
	    const float3 f = _dpd_interaction(dpid, xdest, udest, srcids[NSRCMAX - 1]);
	    
	    xforce += f.x; 
	    yforce += f.y; 
	    zforce += f.z;

	    srccount = NSRCMAX - 1;
	}
    }
    
#pragma unroll 4
    for(int i = 0; i < srccount; ++i)
    {
	const float3 f = _dpd_interaction(dpid, xdest, udest, srcids[i]);
	
	xforce += f.x; 
	yforce += f.y; 
	zforce += f.z;
    }
    
    for(int L = COLS / 2; L > 0; L >>=1)
    {
	xforce += __shfl_xor(xforce, L);
	yforce += __shfl_xor(yforce, L);
	zforce += __shfl_xor(zforce, L);
    }
    
    const float fcontrib = (subtid == 0) * xforce + (subtid == 1) * yforce + (subtid == 2) * zforce;
    
    if (subtid < 3)
	info.axayaz[subtid + 3 * dpid] = fcontrib;
}

template<int COLS, int ROWS, int NSRCMAX>
__device__ void core_ilp(const int nsrc, const int * const scan, const int * const starts, 
		     const int ndst, const int dststart)
{
    const int tid = threadIdx.x; 
    const int slot = tid / COLS;
    const int subtid = tid % COLS;
     
    const int dpid = dststart + slot;
    const int entry = 3 * dpid;
    const float2 dtmp0 = tex1Dfetch(texParticles2, entry);
    const float2 dtmp1 = tex1Dfetch(texParticles2, entry + 1);
    const float2 dtmp2 = tex1Dfetch(texParticles2, entry + 2);
    const float3 xdest = make_float3(dtmp0.x, dtmp0.y, dtmp1.x);
    const float3 udest = make_float3(dtmp1.y, dtmp2.x, dtmp2.y);

    float xforce = 0, yforce = 0, zforce = 0;
    
    for(int s = 0; s < nsrc; s += NSRCMAX * COLS)
    {
	int spids[NSRCMAX];
#pragma unroll 
	for(int i = 0; i < NSRCMAX; ++i)
	{
	    const int pid = s + i * COLS + subtid;
	    const int key9 = 9 * ((pid >= scan[9]) + (pid >= scan[18]));
	    const int key3 = 3 * ((pid >= scan[key9 + 3]) + (pid >= scan[key9 + 6]));
	    const int key = key9 + key3;	    
	    
	    spids[i] = pid - scan[key] + starts[key];
	}

	bool interacting[NSRCMAX];
#pragma unroll 
	for(int i = 0; i < NSRCMAX; ++i)
	{
	    const int sentry = 3 * spids[i];
	    const float2 stmp0 = tex1Dfetch(texParticles2, sentry);
	    const float2 stmp1 = tex1Dfetch(texParticles2, sentry + 1);
	    
	    const float xdiff = xdest.x - stmp0.x;
	    const float ydiff = xdest.y - stmp0.y;
	    const float zdiff = xdest.z - stmp1.x;
	    interacting[i] = (s + i * COLS + subtid < nsrc) && (dpid != spids[i]) && (xdiff * xdiff + ydiff * ydiff + zdiff * zdiff < 1);
	}

#pragma unroll 
	for(int i = 0; i < NSRCMAX; ++i)
	{
	    if (interacting[i])
	    {
		const float3 f = _dpd_interaction(dpid, xdest, udest, spids[i]);
		
		xforce += f.x; 
		yforce += f.y; 
		zforce += f.z;
	    }
	}
    }

    for(int L = COLS / 2; L > 0; L >>=1)
    {
	xforce += __shfl_xor(xforce, L);
	yforce += __shfl_xor(yforce, L);
	zforce += __shfl_xor(zforce, L);
    }
    
    const float fcontrib = (subtid == 0) * xforce + (subtid == 1) * yforce + (subtid == 2) * zforce;
    
    if (subtid < 3)
	info.axayaz[subtid + 3 * dpid] = fcontrib; 
}

#define __IMOD(x,y) ((x)-((x)/(y))*(y))
template<int COLS, int ROWS>
__global__  
__launch_bounds__(32*CPB, 16)
void _dpd_forces_new2() {

	int mycount=0, myscan=0; 

	__shared__ int volatile starts[CPB][32], scan[CPB][32];

	if (threadIdx.x < 14) {

		const int cbase = blockIdx.x*blockDim.y + threadIdx.y;

		int dx, dy, dz;
		dx = dy = dz = threadIdx.x/3;
		dx = threadIdx.x - dx*3 - 1;
		dy = __IMOD(dy,3) - 1;
		dz = __IMOD(dz/3,3) - 1;

		int cid = cbase +
			  dz*info.ncells.x*info.ncells.y +
			  dy*info.ncells.x +
			  dx;

		const bool valid_cid = (cid >= 0) && (cid < info.ncells.x*info.ncells.y*info.ncells.z);

		starts[threadIdx.y][threadIdx.x] = (valid_cid) ? tex1Dfetch(texStart, cid) : 0;
		myscan = mycount = (valid_cid) ? tex1Dfetch(texCount, cid) : 0;
	}
   
	#pragma unroll 
	for(int L = 1; L < 32; L <<= 1)
		myscan += (threadIdx.x >= L)*__shfl_up(myscan, L);

	if (threadIdx.x < 15) scan[threadIdx.y][threadIdx.x] = myscan - mycount;
    
	const int subtid = threadIdx.x % COLS;
	const int slot = threadIdx.x / COLS;

	const int dststart = starts[threadIdx.y][13];
	const int lastdst = dststart + scan[threadIdx.y][14]-scan[threadIdx.y][13];

	const int nsrc = scan[threadIdx.y][14];
	const int nsrcext = scan[threadIdx.y][13];

	for(int pid = subtid; pid < nsrc; pid += COLS) {

		const int key9 = 9*(pid >= scan[threadIdx.y][9]);
		int key3 = 3*(pid >= scan[threadIdx.y][key9 + 3]);
		key3 += (key9 < 9) ? 3*(pid >= scan[threadIdx.y][key9 + 6]) : 0;
		int spid = pid - scan[threadIdx.y][key3+key9] + starts[threadIdx.y][key3+key9];

		const int sentry = 3 * spid;
		const float2 stmp0 = tex1Dfetch(texParticles2, sentry);
		const float2 stmp1 = tex1Dfetch(texParticles2, sentry + 1);
		const float2 stmp2 = tex1Dfetch(texParticles2, sentry + 2);
		const float3 xsrc = make_float3( stmp0.x, stmp0.y, stmp1.x );
		const float3 usrc = make_float3( stmp1.y, stmp2.x, stmp2.y );
		float fx = 0.f, fy = 0.f, fz = 0.f;

		for(int dpid = dststart+slot; dpid < lastdst; dpid += ROWS) {

//			float3 xdest, udest;
			const float2 dtmp0 = tex1Dfetch(texParticles2, 3*dpid);
//			xdest.x = dtmp0.x;
//			xdest.y = dtmp0.y;
			const float2 dtmp1 = tex1Dfetch(texParticles2, 3*dpid +1);
//			xdest.z = dtmp0.x;
//			udest.x = dtmp0.y;
//			udest.y = dtmp0.x;
//			udest.z = dtmp0.y;
			const float3 xdest = make_float3( dtmp0.x, dtmp0.y, dtmp1.x );
			
			const float d2 = (xdest.x-xsrc.x)*(xdest.x-xsrc.x) +
					 (xdest.y-xsrc.y)*(xdest.y-xsrc.y) +
					 (xdest.z-xsrc.z)*(xdest.z-xsrc.z);

			if ((dpid != spid) && (d2 < 1.0f)) {
				const float2 dtmp2 = tex1Dfetch(texParticles2, 3*dpid +2);
			    const float3 udest = make_float3( dtmp1.y, dtmp2.x, dtmp2.y );
				const float3 f = _dpd_interaction(dpid, xdest, udest, xsrc, usrc, spid);

				atomicAdd(info.axayaz + 3*dpid    , f.x);
				atomicAdd(info.axayaz + 3*dpid + 1, f.y);
				atomicAdd(info.axayaz + 3*dpid + 2, f.z);

				if (pid < nsrcext) {
					fx -= f.x;
					fy -= f.y;
					fz -= f.z;
				}
			}
		}
		atomicAdd(info.axayaz + 3*spid    , fx);
		atomicAdd(info.axayaz + 3*spid + 1, fy);
		atomicAdd(info.axayaz + 3*spid + 2, fz);
	}
}

__device__ uint __lanemask_lt() {
	uint mask;
	asm("mov.u32 %0, %lanemask_lt;" : "=r"(mask) );
	return mask;
}

//template<uint SIMD_WIDTH, uint QSIZE>
#define SIMD_WIDTH 16u
#define NSLOT      2u
#if (SIMD_WIDTH*NSLOT!=32)
#pragma error
#endif

__global__ __launch_bounds__(32 * CPB, 16) 
void _dpd_forces_new3()
{
	uint mycount=0, myscan=0;
	const uint tid = threadIdx.x;
	const uint wid = threadIdx.y;

	__shared__ uint volatile starts[CPB][32], scan[CPB][32];
	__shared__ uint volatile queue[CPB][NSLOT][32];

	if (tid < 27) {
		const int cbase = blockIdx.x*blockDim.y + wid;

		int dx, dy, dz;
		dx = dy = dz = tid/3;
		dx = tid - dx*3 - 1;
		dy = __IMOD(dy,3) - 1;
		dz = __IMOD(dz/3,3) - 1;

		int cid = cbase +
			  dz*info.ncells.x*info.ncells.y +
			  dy*info.ncells.x +
			  dx;

		const bool valid_cid = (cid >= 0) && (cid < info.ncells.x*info.ncells.y*info.ncells.z);

		starts[wid][tid] = (valid_cid) ? tex1Dfetch(texStart, cid) : 0;
		myscan = mycount = (valid_cid) ? tex1Dfetch(texCount, cid) : 0;
	}

	// prefix sum
	#pragma unroll
    for(int L = 1; L < 32; L <<= 1)
	myscan += (tid >= L) * __shfl_up(myscan, L) ;
    if (tid < 28) scan[wid][tid] = myscan - mycount;

    const uint dststart = starts[wid][13];
	const uint lastdst = dststart + scan[wid][14]-scan[wid][13];
    const uint nsrc = scan[wid][27];

	const uint lane = tid % SIMD_WIDTH;
	const uint slot = tid / SIMD_WIDTH;
	// TODO
	const uint slotmask = ( ((1<<SIMD_WIDTH)-1) << (slot*SIMD_WIDTH) );
	const uint lanemask = __lanemask_lt() & slotmask;

	for(uint dpid = dststart + slot; dpid < lastdst; dpid += (32/SIMD_WIDTH)) {
		// build neighbor list
		const float2 dtmp0 = tex1Dfetch( texParticles2, dpid * 3     );
		const float2 dtmp1 = tex1Dfetch( texParticles2, dpid * 3 + 1 );
		const float3 xdest = make_float3( dtmp0.x, dtmp0.y, dtmp1.x );
		uint nb = 0;

		for(uint p = 0; p < nsrc; p += SIMD_WIDTH ) {
			const uint pid = p + lane;
			int interacting = 0;
			uint spid;

			if ( pid < nsrc ) {
				const uint key9 = 9 * ((pid >= scan[wid][9       ]) + (pid >= scan[wid][18      ]));
				const uint key3 = 3 * ((pid >= scan[wid][key9 + 3]) + (pid >= scan[wid][key9 + 6]));
				const uint key = key9 + key3;
				spid = pid - scan[wid][key] + starts[wid][key];

				const float2 stmp0 = tex1Dfetch(texParticles2, 3 * spid     );
				const float2 stmp1 = tex1Dfetch(texParticles2, 3 * spid + 1 );
				const float3 xsrc = make_float3( stmp0.x, stmp0.y, stmp1.x );

				const float dx = xdest.x - xsrc.x;
				const float dy = xdest.y - xsrc.y;
				const float dz = xdest.z - xsrc.z;
				const float d2 = dx * dx + dy * dy + dz * dz;

				interacting = ((dpid != spid) && (d2 < 1.0f));
			}
			uint all_interacting = __ballot( interacting );

			uint insert = nb + __popc( all_interacting & lanemask );
			nb += __popc( all_interacting & slotmask );
			if (interacting) queue[wid][slot][insert] = spid;
		}

		// evaluate force
		const float2 dtmp2 = tex1Dfetch( texParticles2, dpid * 3 + 2 );
		const float3 udest = make_float3( dtmp1.y, dtmp2.x, dtmp2.y );

		float xforce = 0.f, yforce = 0.f, zforce = 0.f;
		for(uint pid = lane; pid < nb; pid += SIMD_WIDTH ) {
			const uint spid = queue[wid][slot][pid];
			const float2 stmp0 = tex1Dfetch(texParticles2, 3 * spid     );
			const float2 stmp1 = tex1Dfetch(texParticles2, 3 * spid + 1 );
			const float2 stmp2 = tex1Dfetch(texParticles2, 3 * spid + 2 );
			const float3 xsrc = make_float3( stmp0.x, stmp0.y, stmp1.x );
			const float3 usrc = make_float3( stmp1.y, stmp2.x, stmp2.y );

			const float3 f = _dpd_interaction(dpid, xdest, udest, xsrc, usrc, spid );

			xforce += f.x;
			yforce += f.y;
			zforce += f.z;
		}

		// reduce force
		#pragma unroll
		for(int L = SIMD_WIDTH / 2; L > 0; L >>=1)
		{
			xforce += __shfl_xor(xforce, L);
			yforce += __shfl_xor(yforce, L);
			zforce += __shfl_xor(zforce, L);
		}

		const float fcontrib = (lane == 0) * xforce + (lane == 1) * yforce + (lane == 2) * zforce;
		if (lane < 3u) info.axayaz[3 * dpid + lane] = fcontrib;
	}
}

__global__ __launch_bounds__(32 * CPB, 16)
void _dpd_forces()
{
    assert(blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1);

    const int tid = threadIdx.x;
    const int wid = threadIdx.y;

    __shared__ volatile int starts[CPB][32], scan[CPB][32];

    int mycount = 0, myscan = 0;
    if (tid < 27)
    {
	const int dx = (tid) % 3;
	const int dy = ((tid / 3)) % 3;
	const int dz = ((tid / 9)) % 3;

	int xcid = blockIdx.x * _XCPB_ + ((threadIdx.y) % _XCPB_) + dx - 1;
	int ycid = blockIdx.y * _YCPB_ + ((threadIdx.y / _XCPB_) % _YCPB_) + dy - 1;
	int zcid = blockIdx.z * _ZCPB_ + ((threadIdx.y / (_XCPB_ * _YCPB_)) % _ZCPB_) + dz - 1;

	const bool valid_cid =
	    xcid >= 0 && xcid < info.ncells.x &&
	    ycid >= 0 && ycid < info.ncells.y &&
	    zcid >= 0 && zcid < info.ncells.z ;

	xcid = min(info.ncells.x - 1, max(0, xcid));
	ycid = min(info.ncells.y - 1, max(0, ycid));
	zcid = min(info.ncells.z - 1, max(0, zcid));

	const int cid = max(0, xcid + info.ncells.x * (ycid + info.ncells.y * zcid));

	starts[wid][tid] = tex1Dfetch(texStart, cid);

	myscan = mycount = valid_cid * tex1Dfetch(texCount, cid);
    }

    for(int L = 1; L < 32; L <<= 1)
	myscan += (tid >= L) * __shfl_up(myscan, L) ;

    if (tid < 28)
	scan[wid][tid] = myscan - mycount;

    const int nsrc = scan[wid][27];

    const int dststart = starts[wid][1 + 3 + 9];
    const int ndst = scan[wid][1 + 3 + 9 + 1] - scan[wid][1 + 3 + 9];
    const int ndst4 = (ndst >> 2) << 2;

    for(int d = 0; d < ndst4; d += 4)
	core<8, 4, 4>(nsrc, (const int *)scan[wid], (const int *)starts[wid], 4, dststart + d);
	//core_ilp_test<32, 1, 4>(nsrc, (const int *)scan[wid], (const int *)starts[wid], 4, dststart + d);

    int d = ndst4;
    if (d + 2 <= ndst)
    {
	core<16, 2, 4>(nsrc, (const int *)scan[wid],  (const int *)starts[wid], 2, dststart + d);
	d += 2;
    }

    if (d < ndst)
	core_ilp<32, 1, 2>(nsrc, (const int *)scan[wid], (const int *)starts[wid], 1, dststart + d);
}

#else
__global__ __launch_bounds__(32 * CPB, 16) 
    void _dpd_forces()
{
    const int COLS = 32;
    const int ROWS = 1;
    assert(warpSize == COLS * ROWS);
    assert(blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1);
    assert(ROWS * 3 <= warpSize);

    const int tid = threadIdx.x; 
    const int subtid = tid % COLS;
    const int slot = tid / COLS;
    const int wid = threadIdx.y;
     
    __shared__ int volatile starts[CPB][32], scan[CPB][32];

    int mycount = 0, myscan = 0; 
    if (tid < 27)
    {
	const int dx = (tid) % 3;
	const int dy = ((tid / 3)) % 3; 
	const int dz = ((tid / 9)) % 3;

	int xcid = blockIdx.x * _XCPB_ + ((threadIdx.y) % _XCPB_) + dx - 1;
	int ycid = blockIdx.y * _YCPB_ + ((threadIdx.y / _XCPB_) % _YCPB_) + dy - 1;
	int zcid = blockIdx.z * _ZCPB_ + ((threadIdx.y / (_XCPB_ * _YCPB_)) % _ZCPB_) + dz - 1;
	
	const bool valid_cid = 
	    xcid >= 0 && xcid < info.ncells.x &&
	    ycid >= 0 && ycid < info.ncells.y &&
	    zcid >= 0 && zcid < info.ncells.z ;
	
	xcid = min(info.ncells.x - 1, max(0, xcid));
	ycid = min(info.ncells.y - 1, max(0, ycid));
	zcid = min(info.ncells.z - 1, max(0, zcid));
	
	const int cid = max(0, xcid + info.ncells.x * (ycid + info.ncells.y * zcid));
	
	starts[wid][tid] = tex1Dfetch(texStart, cid);
	
	myscan = mycount = valid_cid * tex1Dfetch(texCount, cid);
    }

    for(int L = 1; L < 32; L <<= 1)
	myscan += (tid >= L) * __shfl_up(myscan, L) ;

    if (tid < 28)
	scan[wid][tid] = myscan - mycount;

    const int dststart = starts[wid][1 + 3 + 9];
    const int nsrc = scan[wid][27], ndst = scan[wid][1 + 3 + 9 + 1] - scan[wid][1 + 3 + 9];
 
    for(int d = 0; d < ndst; d += ROWS)
    {
	const int np1 = min(ndst - d, ROWS);

	const int dpid = dststart + d + slot;
	const int entry = 3 * dpid;
	// Mauro: redundant reads by the warp (ROW S=1 && COLS=32 -> the 32 th of the warp read the same float2 x3)
	float2 dtmp0 = tex1Dfetch(texParticles2, entry);
	float2 dtmp1 = tex1Dfetch(texParticles2, entry + 1);
	float2 dtmp2 = tex1Dfetch(texParticles2, entry + 2);
	
	float xforce = 0, yforce = 0, zforce = 0;

	for(int s = 0; s < nsrc; s += COLS)
	{
	    const int np2 = min(nsrc - s, COLS);
  
	    const int pid = s + subtid;
	    const int key9 = 9 * ((pid >= scan[wid][9]) + (pid >= scan[wid][18]));
	    const int key3 = 3 * ((pid >= scan[wid][key9 + 3]) + (pid >= scan[wid][key9 + 6]));
	    const int key = key9 + key3;	    
	   
	    const int spid = pid - scan[wid][key] + starts[wid][key];
	    const int sentry = 3 * spid;
	    const float2 stmp0 = tex1Dfetch(texParticles2, sentry);
	    const float2 stmp1 = tex1Dfetch(texParticles2, sentry + 1);
	    const float2 stmp2 = tex1Dfetch(texParticles2, sentry + 2);

#ifndef NDEBUG
	    {
		const int key1 = (pid >= scan[wid][key9 + key3 + 1]) + (pid >= scan[wid][key9 + key3 + 2]);
		const int keyref = key9 + key3 + key1;
		assert(keyref >= 0 && keyref < 27);
		assert(pid >= scan[wid][keyref]);
		assert(pid < scan[wid][keyref + 1] || pid >= nsrc);

		const int spidref = pid - scan[wid][keyref] + starts[wid][keyref];
		assert(spidref == spid || pid >= nsrc);
	    }
#endif
	    
	    {
		const float xdiff = dtmp0.x - stmp0.x;
		const float ydiff = dtmp0.y - stmp0.y;
		const float zdiff = dtmp1.x - stmp1.x;

#ifndef _NONPERIODIC_KERNEL_
		asdasda
		const float _xr = xdiff - info.domainsize.x * floorf(0.5f + xdiff * info.invdomainsize.x);
		const float _yr = ydiff - info.domainsize.y * floorf(0.5f + ydiff * info.invdomainsize.y);
		const float _zr = zdiff - info.domainsize.z * floorf(0.5f + zdiff * info.invdomainsize.z);
#else
		const float _xr = xdiff;
		const float _yr = ydiff;
		const float _zr = zdiff;
#endif
		const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
		const float invrij = rsqrtf(rij2);
		const float rij = rij2 * invrij;
		const float argwr = max((float)0, 1 - rij * info.invrc);
		const float wr = viscosity_function<-VISCOSITY_S_LEVEL>(argwr);
		const float xr = _xr * invrij;
		const float yr = _yr * invrij;
		const float zr = _zr * invrij;
		
		const float rdotv = 
		    xr * (dtmp1.y - stmp1.y) +
		    yr * (dtmp2.x - stmp2.x) +
		    zr * (dtmp2.y - stmp2.y);
		  
		const float myrandnr = Logistic::mean0var1(info.seed, min(spid, dpid), max(spid, dpid));

		const float strength = info.aij * argwr - (info.gamma * wr * rdotv + info.sigmaf * myrandnr) * wr;
		const bool valid = (dpid != spid) && (slot < np1) && (subtid < np2);
		
		if (valid)
		{
#ifdef _CHECK_
		    xforce += (rij2 < 1);
		    yforce += wr;
		    zforce += 0;
#else		    	     
		    xforce += strength * xr;
		    yforce += strength * yr;
		    zforce += strength * zr;
#endif
		}
	    } 
	}
	
	for(int L = COLS / 2; L > 0; L >>=1)
	{
	    xforce += __shfl_xor(xforce, L);
	    yforce += __shfl_xor(yforce, L);
	    zforce += __shfl_xor(zforce, L);
	}

	const int c = (subtid % 3);       
	const float fcontrib = (c == 0) * xforce + (c == 1) * yforce + (c == 2) * zforce;//f[subtid % 3];
	const int dstpid = dststart + d + slot;

	if (slot < np1)
	    info.axayaz[c + 3 * dstpid] = fcontrib;
    }
}
#endif


#ifdef _INSPECT_
__global__ __launch_bounds__(32 * CPB, 8) 
    void inspect_dpd_forces(const int COLS, const int ROWS, const int nparticles, int2 * const entries, const int nentries)
{
    assert(nentries = COLS * nparticles);
    assert(warpSize == COLS * ROWS);
    assert(blockDim.x == warpSize && blockDim.y == CPB && blockDim.z == 1);
    assert(ROWS * 3 <= warpSize);

    const int tid = threadIdx.x; 
    const int subtid = tid % COLS;
    const int slot = tid / COLS;
    const int wid = threadIdx.y;
 
    __shared__ int volatile starts[CPB][32], scan[CPB][32];

    int mycount = 0, myscan = 0; 
    if (tid < 27)
    {
	const int dx = (tid) % 3;
	const int dy = ((tid / 3)) % 3; 
	const int dz = ((tid / 9)) % 3;

	int xcid = blockIdx.x * _XCPB_ + ((threadIdx.y) % _XCPB_) + dx - 1;
	int ycid = blockIdx.y * _YCPB_ + ((threadIdx.y / _XCPB_) % _YCPB_) + dy - 1;
	int zcid = blockIdx.z * _ZCPB_ + ((threadIdx.y / (_XCPB_ * _YCPB_)) % _ZCPB_) + dz - 1;
	
	const bool valid_cid = 
	    xcid >= 0 && xcid < info.ncells.x &&
	    ycid >= 0 && ycid < info.ncells.y &&
	    zcid >= 0 && zcid < info.ncells.z ;
	
	xcid = min(info.ncells.x - 1, max(0, xcid));
	ycid = min(info.ncells.y - 1, max(0, ycid));
	zcid = min(info.ncells.z - 1, max(0, zcid));
	
	const int cid = max(0, xcid + info.ncells.x * (ycid + info.ncells.y * zcid));
	
	starts[wid][tid] = tex1Dfetch(texStart, cid);
	
	myscan = mycount = valid_cid * tex1Dfetch(texCount, cid);
    }

    for(int L = 1; L < 32; L <<= 1)
	myscan += (tid >= L) * __shfl_up(myscan, L) ;

    if (tid < 28)
	scan[wid][tid] = myscan - mycount;

    const int dststart = starts[wid][1 + 3 + 9];
    const int nsrc = scan[wid][27], ndst = scan[wid][1 + 3 + 9 + 1] - scan[wid][1 + 3 + 9];
 
    for(int d = 0; d < ndst; d += ROWS)
    {
	//int srccount = 0;
	
	const int np1 = min(ndst - d, ROWS);

	const int dpid = dststart + d + slot;
	const int entry = 3 * dpid;
	
	const float2 dtmp0 = tex1Dfetch(texParticles2, entry);
	const float2 dtmp1 = tex1Dfetch(texParticles2, entry + 1);
	const float2 dtmp2 = tex1Dfetch(texParticles2, entry + 2);
	const float3 xdest = make_float3(dtmp0.x, dtmp0.y, dtmp1.x);
	const float3 udest = make_float3(dtmp1.y, dtmp2.x, dtmp2.y);
	
	int ninteractions = 0, npotentialinteractions = 0;
	
	for(int s = 0; s < nsrc; s += COLS)
	{
	    const int np2 = min(nsrc - s, COLS);
  
	    const int pid = s + subtid;
	    const int key9 = 9 * ((pid >= scan[wid][9]) + (pid >= scan[wid][18]));
	    const int key3 = 3 * ((pid >= scan[wid][key9 + 3]) + (pid >= scan[wid][key9 + 6]));
	    const int key = key9 + key3;	    
	   
	    const int spid = pid - scan[wid][key] + starts[wid][key];
	    const int sentry = 3 * spid;
	    const float2 stmp0 = tex1Dfetch(texParticles2, sentry);
	    const float2 stmp1 = tex1Dfetch(texParticles2, sentry + 1);

	    const float xdiff = xdest.x - stmp0.x;
	    const float ydiff = xdest.y - stmp0.y;
	    const float zdiff = xdest.z - stmp1.x;
	    const bool interacting = (dpid != spid) && (slot < np1) && (subtid < np2) &&
		(xdiff * xdiff + ydiff * ydiff + zdiff * zdiff < 1);
    
	    ninteractions += (int)(interacting);
	    npotentialinteractions += 1;
	}

	if (slot < np1)
	    entries[subtid + COLS * dpid] = make_int2(ninteractions, npotentialinteractions);
    }
}
#endif

bool fdpd_init = false;

#include "../hacks.h"
#ifdef _TIME_PROFILE_
static cudaEvent_t evstart, evstop;
#endif

static void cellstats(const int *d_cstart, const int *d_ccount, int nx, int ny, int nz) {

	int *cstart, *ccount;
	int cmin=999999999, cmax=-1, nmin=999999999, nmax=-1;
	int nc = nx*ny*nz;

	cstart = (int *)malloc(sizeof(*cstart)*nc);
	if (!cstart) {
		fprintf(stderr, "Cannot allocate %zu bytes!\n", sizeof(*cstart)*nc);
		exit(1);
	}
	ccount = (int *)malloc(sizeof(*ccount)*nc);
	if (!ccount) {
		fprintf(stderr, "Cannot allocate %zu bytes!\n", sizeof(*ccount)*nc);
		exit(1);
	}

	CUDA_CHECK(cudaMemcpy(cstart, d_cstart, sizeof(*cstart)*nc, cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(ccount, d_ccount, sizeof(*ccount)*nc, cudaMemcpyDeviceToHost));

	for(int i = 0; i < nc; i++) {
		cmin = min(cmin, ccount[i]);
		cmax = max(cmax, ccount[i]);
	}

	int neighs[27];
	for(int k = -1; k < 2; k++)
		for(int j = -1; j < 2; j++)
			for(int i = -1; i < 2; i++)
				neighs[(k+1)*9 + (j+1)*3 + i+1] = k*nx*ny + j*nx + i;

	for(int i = 0; i < nc; i++) {
		int curnn = 0;
		for(int z = 0; z < 27; z++) {
			int nid = i+neighs[z];
			if (nid >= 0 && nid < nc) {
				curnn += ccount[nid];
			}
		}
		nmin = min(nmin, curnn);
		nmax = max(nmax, curnn);
	}

	fprintf(stdout, "Cell particle count min/max: %d, %d\n", cmin, cmax);
	fprintf(stdout, "Cell neighborhood particle count min/max: %d, %d\n", nmin, nmax);

	free(cstart);
	free(ccount);
}

__global__ void print_a(const int np) {
	double sx = 0, sy = 0, sz = 0;
	for(int i=0;i<np;i++) {
//		printf("%f %f %f\n",info.axayaz[i*3+0], info.axayaz[i*3+1], info.axayaz[i*3+2]);
		sx += info.axayaz[i*3+0];
		sy += info.axayaz[i*3+1];
		sz += info.axayaz[i*3+2];
	}
	printf("%lf %lf %lf\n",sx,sy,sz);
}

void forces_dpd_cuda_nohost(const float * const xyzuvw, float * const axayaz,  const int np,
			    const int * const cellsstart, const int * const cellscount, 
			    const float rc,
			    const float XL, const float YL, const float ZL,
			    const float aij,
			    const float gamma,
			    const float sigma,
			    const float invsqrtdt,
			    const float seed, cudaStream_t stream)
{
    if (np == 0)
    {
	printf("WARNING: forces_dpd_cuda_nohost called with np = %d\n", np);
	return;
    }

    int nx = (int)ceil(XL / rc);
    int ny = (int)ceil(YL / rc);
    int nz = (int)ceil(ZL / rc);
    const int ncells = nx * ny * nz;

    if (!fdpd_init)
    {
	texStart.channelDesc = cudaCreateChannelDesc<int>();
	texStart.filterMode = cudaFilterModePoint;
	texStart.mipmapFilterMode = cudaFilterModePoint;
	texStart.normalized = 0;
    
	texCount.channelDesc = cudaCreateChannelDesc<int>();
	texCount.filterMode = cudaFilterModePoint;
	texCount.mipmapFilterMode = cudaFilterModePoint;
	texCount.normalized = 0;

	texParticles2.channelDesc = cudaCreateChannelDesc<float2>();
	texParticles2.filterMode = cudaFilterModePoint;
	texParticles2.mipmapFilterMode = cudaFilterModePoint;
	texParticles2.normalized = 0;

	//void (*dpdkernel)() =  _dpd_forces;
	void (*dpdkernel)() =  _dpd_forces_new3;//, 3>;

	CUDA_CHECK(cudaFuncSetCacheConfig(*dpdkernel, cudaFuncCachePreferEqual));

#ifdef _TIME_PROFILE_
	CUDA_CHECK(cudaEventCreate(&evstart));
	CUDA_CHECK(cudaEventCreate(&evstop));
#endif
	fdpd_init = true;
    }

    size_t textureoffset;
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texParticles2, xyzuvw, &texParticles2.channelDesc, sizeof(float) * 6 * np));
    assert(textureoffset == 0);
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texStart, cellsstart, &texStart.channelDesc, sizeof(int) * ncells));
    assert(textureoffset == 0);
    CUDA_CHECK(cudaBindTexture(&textureoffset, &texCount, cellscount, &texCount.channelDesc, sizeof(int) * ncells));
    assert(textureoffset == 0);
      
    InfoDPD c;
    c.ncells = make_int3(nx, ny, nz);
    c.domainsize = make_float3(XL, YL, ZL);
    c.invdomainsize = make_float3(1 / XL, 1 / YL, 1 / ZL);
    c.domainstart = make_float3(-XL * 0.5, -YL * 0.5, -ZL * 0.5);
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
    c.axayaz = axayaz;
    c.seed = seed;
      
    CUDA_CHECK(cudaMemcpyToSymbolAsync(info, &c, sizeof(c), 0, cudaMemcpyHostToDevice, stream));
   
    static int cetriolo = 0;
    cetriolo++;

#ifdef _INSPECT_
    {
	//inspect irregularity of the computation,
	//report data to file
	if (cetriolo % 1000 == 0)
	{
	    enum { COLS = 16, ROWS = 2 };

	    const size_t nentries = np * COLS;

	    int2 * data;
	    CUDA_CHECK(cudaHostAlloc(&data, sizeof(int2) * nentries, cudaHostAllocMapped));
	    memset(data, 0xff, sizeof(int2) * nentries);
	    
	    int * devptr;
	    CUDA_CHECK(cudaHostGetDevicePointer(&devptr, data, 0));

	    inspect_dpd_forces<<<dim3(c.ncells.x / _XCPB_, c.ncells.y / _YCPB_, c.ncells.z / _ZCPB_), dim3(32, CPB), 0, stream>>>
		(COLS, ROWS, np, data, nentries);

	    CUDA_CHECK(cudaDeviceSynchronize());

	    char path2report[2000];
	    sprintf(path2report, "inspection-%d-tstep.txt", cetriolo);

	    FILE * f = fopen(path2report, "w");
	    assert(f);

	    for(int i = 0, c = 0; i < np; ++i)
	    {
		fprintf(f, "pid %05d: ", i);
		
		int s = 0, pot = 0;
		for(int j = 0; j < COLS; ++j, ++c)
		{
		    fprintf(f, "%02d ", data[c].x);
		    s += data[c].x;
		    pot += data[c].y;
		}
		
		fprintf(f, " sum: %02d pot: %d\n", s, (pot + COLS - 1) / (COLS));
	    }
	    
	    fclose(f);
	    
	    CUDA_CHECK(cudaFreeHost(data));
	    printf("inspection saved to %s.\n", path2report);
	}
    }
#endif

#ifdef _TIME_PROFILE_
    if (cetriolo % 500 == 0)
	CUDA_CHECK(cudaEventRecord(evstart));
#endif
//fprintf("Calling _dpd_forces with conf g(%d,%d,%d), b (%d,%d,%d)\n", c.ncells.x / _XCPB_, c.ncells.y / _YCPB_, c.ncells.z / _ZCPB_, 32, CPB, 1);

    //cellstats(cellsstart, cellscount, nx, ny, nz);              // MAURO
    CUDA_CHECK(cudaMemset(axayaz, 0, sizeof(float)*np*3));      /////////// MAURO CHECK IF NECESSARY

    _dpd_forces_new3/*, 3>*/<<<(c.ncells.x*c.ncells.y*c.ncells.z+CPB-1)/CPB, dim3(32, CPB), 0, stream>>>();
    //_dpd_forces<<<dim3(c.ncells.x / _XCPB_, c.ncells.y / _YCPB_, c.ncells.z / _ZCPB_), dim3(32, CPB), 0, stream>>>();

#ifdef ONESTEP
	print_a<<<1,1>>>(np);

    CUDA_CHECK( cudaDeviceSynchronize() );
    CUDA_CHECK( cudaDeviceReset() );
    printf("%s %d\n",__FILE__,__LINE__);
    exit(0);
#endif

#ifdef _TIME_PROFILE_
    if (cetriolo % 500 == 0)
    {
	CUDA_CHECK(cudaEventRecord(evstop));
	CUDA_CHECK(cudaEventSynchronize(evstop));
	
	float tms;
	CUDA_CHECK(cudaEventElapsedTime(&tms, evstart, evstop));
	printf("elapsed time for DPD-BULK kernel: %.2f ms\n", tms);
    }
#endif

    CUDA_CHECK(cudaPeekAtLastError());	
}

#include <cmath>
#include <unistd.h>

#include "../cell-lists.h"

int fdpd_oldnp = 0, fdpd_oldnc = 0;

float * fdpd_xyzuvw = NULL, * fdpd_axayaz = NULL;
int * fdpd_start = NULL, * fdpd_count = NULL;


void forces_dpd_cuda_aos(float * const _xyzuvw, float * const _axayaz,
		     int * const order, const int np,
		     const float rc,
		     const float XL, const float YL, const float ZL,
		     const float aij,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt,
			 const float seed,
			 const bool nohost)
{
    if (np == 0)
    {
	printf("WARNING: forces_dpd_cuda_aos called with np = %d\n", np);
	return;
    }

    int nx = (int)ceil(XL / rc);
    int ny = (int)ceil(YL / rc);
    int nz = (int)ceil(ZL / rc);
    const int ncells = nx * ny * nz;

    if (!fdpd_init)
    {
	texStart.channelDesc = cudaCreateChannelDesc<int>();
	texStart.filterMode = cudaFilterModePoint;
	texStart.mipmapFilterMode = cudaFilterModePoint;
	texStart.normalized = 0;
    
	texCount.channelDesc = cudaCreateChannelDesc<int>();
	texCount.filterMode = cudaFilterModePoint;
	texCount.mipmapFilterMode = cudaFilterModePoint;
	texCount.normalized = 0;

	texParticles2.channelDesc = cudaCreateChannelDesc<float2>();
	texParticles2.filterMode = cudaFilterModePoint;
	texParticles2.mipmapFilterMode = cudaFilterModePoint;
	texParticles2.normalized = 0;

	fdpd_init = true;
    }
    
    if (fdpd_oldnp < np)
    {
	if (fdpd_oldnp > 0)
	{
	    CUDA_CHECK(cudaFree(fdpd_xyzuvw));
	    CUDA_CHECK(cudaFree(fdpd_axayaz));
	}

	CUDA_CHECK(cudaMalloc(&fdpd_xyzuvw, sizeof(float) * 6 * np));
	CUDA_CHECK(cudaMalloc(&fdpd_axayaz, sizeof(float) * 3 * np));

	size_t textureoffset;
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texParticles2, fdpd_xyzuvw, &texParticles2.channelDesc, sizeof(float) * 6 * np));
	
	fdpd_oldnp = np;
    }

    if (fdpd_oldnc < ncells)
    {
	if (fdpd_oldnc > 0)
	{
	    CUDA_CHECK(cudaFree(fdpd_start));
	    CUDA_CHECK(cudaFree(fdpd_count));
	}

	CUDA_CHECK(cudaMalloc(&fdpd_start, sizeof(int) * ncells));
	CUDA_CHECK(cudaMalloc(&fdpd_count, sizeof(int) * ncells));

	size_t textureoffset = 0;
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texStart, fdpd_start, &texStart.channelDesc, sizeof(int) * ncells));
	CUDA_CHECK(cudaBindTexture(&textureoffset, &texCount, fdpd_count, &texCount.channelDesc, sizeof(int) * ncells));
	
	fdpd_oldnc = ncells;
    }

    CUDA_CHECK(cudaMemcpyAsync(fdpd_xyzuvw, _xyzuvw, sizeof(float) * np * 6, nohost ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice, 0));
    
    InfoDPD c;
    c.ncells = make_int3(nx, ny, nz);
    c.domainsize = make_float3(XL, YL, ZL);
    c.invdomainsize = make_float3(1 / XL, 1 / YL, 1 / ZL);
    c.domainstart = make_float3(-XL * 0.5, -YL * 0.5, -ZL * 0.5);
    c.invrc = 1.f / rc;
    c.aij = aij;
    c.gamma = gamma;
    c.sigmaf = sigma * invsqrtdt;
    c.axayaz = fdpd_axayaz;
    c.seed = seed;

    build_clists(fdpd_xyzuvw, np, rc, c.ncells.x, c.ncells.y, c.ncells.z,
		 c.domainstart.x, c.domainstart.y, c.domainstart.z,
		 order, fdpd_start, fdpd_count, NULL);

    //TextureWrap texStart(_ptr(starts), ncells), texCount(_ptr(counts), ncells);
    //TextureWrap texParticles((float2*)_ptr(xyzuvw), 3 * np);
    
    CUDA_CHECK(cudaMemcpyToSymbolAsync(info, &c, sizeof(c), 0));

    //cellstats(fdpd_start, fdpd_count, nx, ny, nz);              // MAURO
    CUDA_CHECK(cudaMemset(fdpd_axayaz, 0, sizeof(float)*np*3)); ////////////// Mauro: CHECK IF NECESSARY 

    _dpd_forces_new3<<<(c.ncells.x*c.ncells.y*c.ncells.z+CPB-1)/CPB, dim3(32, CPB)>>>();
    //_dpd_forces<<<dim3(c.ncells.x / _XCPB_, c.ncells.y / _YCPB_, c.ncells.z / _ZCPB_), dim3(32, CPB)>>>();
 
    CUDA_CHECK(cudaPeekAtLastError());
	
//copy xyzuvw as well?!?
    if (nohost)
    {
	CUDA_CHECK(cudaMemcpy(_xyzuvw, fdpd_xyzuvw, sizeof(float) * 6 * np, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(_axayaz, fdpd_axayaz, sizeof(float) * 3 * np, cudaMemcpyDeviceToDevice));
    }
    else
	CUDA_CHECK(cudaMemcpy(_axayaz, fdpd_axayaz, sizeof(float) * 3 * np, cudaMemcpyDeviceToHost));

#ifdef _CHECK_
    CUDA_CHECK(cudaThreadSynchronize());
    
    for(int ii = 0; ii < np; ++ii)
    { 
	printf("pid %d -> %f %f %f\n", ii, (float)axayaz[0 + 3 * ii], (float)axayaz[1 + 3* ii], (float)axayaz[2 + 3 *ii]);

	int cnt = 0;
	float fc = 0;
	const int i = order[ii];
	printf("devi coords are %f %f %f\n", (float)xyzuvw[0 + 6 * ii], (float)xyzuvw[1 + 6 * ii], (float)xyzuvw[2 + 6 * ii]);
	printf("host coords are %f %f %f\n", (float)_xyzuvw[0 + 6 * i], (float)_xyzuvw[1 + 6 * i], (float)_xyzuvw[2 + 6 * i]);
	
	for(int j = 0; j < np; ++j)
	{
	    if (i == j) 
		continue;
 
	    float xr = _xyzuvw[0 + 6 *i] - _xyzuvw[0 + 6 * j];
	    float yr = _xyzuvw[1 + 6 *i] - _xyzuvw[1 + 6 * j];
	    float zr = _xyzuvw[2 + 6 *i] - _xyzuvw[2 + 6 * j];

	    xr -= c.domainsize.x *  ::floor(0.5f + xr / c.domainsize.x);
	    yr -= c.domainsize.y *  ::floor(0.5f + yr / c.domainsize.y);
	    zr -= c.domainsize.z *  ::floor(0.5f + zr / c.domainsize.z);

	    const float rij2 = xr * xr + yr * yr + zr * zr;
	    const float invrij = rsqrtf(rij2);
	    const float rij = rij2 * invrij;
	    const float wr = max((float)0, 1 - rij * c.invrc);
	
	    const bool collision =  rij2 < 1;

	    if (collision)
		fc += wr;//	printf("ref p %d colliding with %d\n", i, j);
	    
	    cnt += collision;
	}
	printf("i found %d host interactions and with cuda i found %d\n", cnt, (int)axayaz[0 + 3 * ii]);
	assert(cnt == (float)axayaz[0 + 3 * ii]);
	printf("fc aij ref %f vs cuda %e\n", fc,  (float)axayaz[1 + 3 * ii]);
	assert(fabs(fc - (float)axayaz[1 + 3 * ii]) < 1e-4);
    }
    
    printf("test done.\n");
    sleep(1);
    exit(0);
#endif
}


int * fdpd_order = NULL;
float * fdpd_pv = NULL, *fdpd_a = NULL;

void forces_dpd_cuda(const float * const xp, const float * const yp, const float * const zp,
		     const float * const xv, const float * const yv, const float * const zv,
		     float * const xa, float * const ya, float * const za,
		     const int np,
		     const float rc,
		     const float LX, const float LY, const float LZ,
		     const float aij,
		     const float gamma,
		     const float sigma,
		     const float invsqrtdt,
		     const float seed)
{
    if (np <= 0) return;

    if (np > fdpd_oldnp)
    {
	if (fdpd_oldnp > 0)
	{
	    CUDA_CHECK(cudaFreeHost(fdpd_pv));
	    CUDA_CHECK(cudaFreeHost(fdpd_order));
	    CUDA_CHECK(cudaFreeHost(fdpd_a));
	}

	CUDA_CHECK(cudaHostAlloc(&fdpd_pv, sizeof(float) * np * 6, cudaHostAllocDefault));
	CUDA_CHECK(cudaHostAlloc(&fdpd_order, sizeof(int) * np, cudaHostAllocDefault));
	CUDA_CHECK(cudaHostAlloc(&fdpd_a, sizeof(float) * np * 3, cudaHostAllocDefault));

	//this will be done by forces_dpd_cuda
	//fdpd_oldnp = np;
    }
    
    for(int i = 0; i < np; ++i)
    {
	fdpd_pv[0 + 6 * i] = xp[i];
	fdpd_pv[1 + 6 * i] = yp[i];
	fdpd_pv[2 + 6 * i] = zp[i];
	fdpd_pv[3 + 6 * i] = xv[i];
	fdpd_pv[4 + 6 * i] = yv[i];
	fdpd_pv[5 + 6 * i] = zv[i];
    }

    forces_dpd_cuda_aos(fdpd_pv, fdpd_a, fdpd_order, np, rc, LX, LY, LZ,
			aij, gamma, sigma, invsqrtdt, seed, false);
    
    //delete [] pv;
     
    for(int i = 0; i < np; ++i)
    {
	xa[fdpd_order[i]] += fdpd_a[0 + 3 * i];
	ya[fdpd_order[i]] += fdpd_a[1 + 3 * i];
	za[fdpd_order[i]] += fdpd_a[2 + 3 * i];
    }

    //delete [] a;

    //delete [] order;
}
