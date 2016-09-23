#include <cstdio>
#include <helper_math.h>
#include <cuda_fp16.h>
#include <cassert>
#include "../tiny-float.h"

#include "cuda-dpd.h"
#include "../dpd-rng.h"

__device__ __forceinline__ float3 readCoosFromAll4(const float4* xyzouvwo, int pid)
{
	const float4 tmp = xyzouvwo[2*pid];

	return make_float3(tmp.x, tmp.y, tmp.z);
}

__device__ __forceinline__ float3 readCoos4(const float4* xyzo, int pid)
{
	const float4 tmp = xyzo[pid];

	return make_float3(tmp.x, tmp.y, tmp.z);
}

__device__ __forceinline__ float3 readCoos2(const float2* xyzuvw, int pid)
{
	const float2 tmp1 = xyzuvw[pid*3];
	const float2 tmp2 = xyzuvw[pid*3+1];

	return make_float3(tmp1.x, tmp1.y, tmp2.x);
}

__device__ __forceinline__ void readAll4(const float4* xyzouvwo, int pid, float3& coo, float3& vel)
{
	const float4 tmp1 = xyzouvwo[pid*2];
	const float4 tmp2 = xyzouvwo[pid*2+1];

	coo = make_float3(tmp1.x, tmp1.y, tmp1.z);
	vel = make_float3(tmp2.x, tmp2.y, tmp2.z);
}

__device__ __forceinline__ void readAll2(const float2* xyzuvw, int pid, float3& coo, float3& vel)
{
	const float2 tmp1 = xyzuvw[pid*3];
	const float2 tmp2 = xyzuvw[pid*3+1];
	const float2 tmp3 = xyzuvw[pid*3+2];

	coo = make_float3(tmp1.x, tmp1.y, tmp2.x);
	vel = make_float3(tmp2.y, tmp3.x, tmp3.y);
}

__device__ __forceinline__ float sqr(float x)
{
	return x*x;
}

template<typename Ta, typename Tb>
__device__ __forceinline__ float distance2(const Ta a, const Tb b)
{
	return sqr(a.x - b.x) + sqr(a.y - b.y) + sqr(a.z - b.z);
}

template<typename T>
__device__ __forceinline__ float3 _dpd_interaction(const T* __restrict__ data, const int dstId, const int srcId,
		float adpd, float gammadpd, float sigmadpd, float seed)
{
	float3 dstCoo, dstVel, srcCoo, srcVel;
	readAll4(data, dstId, dstCoo, dstVel);
	readAll4(data, srcId, srcCoo, srcVel);

	const float _xr = dstCoo.x - srcCoo.x;
	const float _yr = dstCoo.y - srcCoo.y;
	const float _zr = dstCoo.z - srcCoo.z;
	const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	if (rij2 > 1.0f) return make_float3(0.0f);

	const float invrij = rsqrtf(rij2);
	const float rij = rij2 * invrij;
	const float argwr = 1.0f - rij;
	const float wr = viscosity_function<0>(argwr);

	const float xr = _xr * invrij;
	const float yr = _yr * invrij;
	const float zr = _zr * invrij;

	const float rdotv =
			xr * (dstVel.x - srcVel.x) +
			yr * (dstVel.y - srcVel.y) +
			zr * (dstVel.z - srcVel.z);

	const float myrandnr = 0*Logistic::mean0var1(seed, min(srcId, dstId), max(srcId, dstId));

	const float strength = adpd * argwr - (gammadpd * wr * rdotv + sigmadpd * myrandnr) * wr;

	return make_float3(strength * xr, strength * yr, strength * zr);
}

__device__ __forceinline__ int warpAggrFetchAdd(int& val, bool pred, int tid)
{
	const int mask = ballot(pred); // i-th bit is if lane i is interacting
	int res = val + __popc( mask & ((1 << tid) - 1) );
	val = val + __popc(mask);
	return res;
}


__device__ __forceinline__ int pack2int(int small, int big)
{
	return (small << 26) | big;
}

__device__ __forceinline__ int2 upackFromint(int packed, int shift)
{
	return make_int2( (packed >> 26) + shift,  packed & 0x3FFFFFF );
}

template<typename T>
__device__ __forceinline__ void execInteractions(const int nentries, const int2 pids, const T* data, float* axayaz,
		float adpd, float gammadpd, float sigmadpd, float seed)
{
	//assert(nentries <= 32);

	if (pids.x == pids.y) return;
	float3 frc = _dpd_interaction(data, pids.x, pids.y, adpd, gammadpd, sigmadpd, seed);
	//
	float* dest = axayaz + pids.x*3;
	atomicAdd(dest,     frc.x);
	atomicAdd(dest + 1, frc.y);
	atomicAdd(dest + 2, frc.z);

	// If src is in the same cell as dst,
	// interaction will be computed twice
	// no need for atomics in this case
	dest = axayaz + pids.y*3;
	atomicAdd(dest,     -frc.x);
	atomicAdd(dest + 1, -frc.y);
	atomicAdd(dest + 2, -frc.z);
}

const int bDimY = 2;
const int bDimZ = 2;
const int nDstPerIter = 2;

__launch_bounds__(32*bDimY*bDimZ, 64 / (bDimY*bDimZ))
__global__ void smth(const float2 * __restrict__ xyzuvw, const float4 * __restrict__ xyzo, const float4 * __restrict__ xyzouvwo, float* axayaz, const int * __restrict__ cellsstart,
		int nCellsX, int nCellsY, int nCellsZ, int ncells_1, float adpd, float gammadpd, float sigmadpd, float seed)
{
	const int tid = threadIdx.x;
	const int wid = (threadIdx.z * bDimZ + threadIdx.y);

	volatile __shared__ int pool[ bDimY*bDimZ ][128];
	volatile int* inters = pool[wid];
	int nentries = 0;

	const int cellX0 = (blockIdx.x) ;
	const int cellY0 = (blockIdx.y * bDimY) + threadIdx.y;
	const int cellZ0 = (blockIdx.z * bDimZ) + threadIdx.z;

	const int cid = (cellZ0*nCellsY + cellY0)*nCellsX + cellX0;

	const int dstStart = cellsstart[cid];
	const int dstEnd   = cellsstart[cid+1];

	const int groupSize = 6;
	const int groupId = (uint)tid / groupSize;//tid / groupSize;
	const int idInGroup = tid - groupId*6;//groupSize * groupId;

	int shZ = (uint)groupId / 3;
	int cellY = cellY0 + (groupId - shZ*3) - 1;
	int cellZ = cellZ0 + shZ - 1;

	bool valid = (cellY >= 0 && cellY < nCellsY && cellZ >= 0 && cellZ < nCellsZ && groupId < 5);

	const int midCellId = (cellZ*nCellsY + cellY)*nCellsX + cellX0;
	int rowStart  = max(midCellId-1, 0);
	int rowEnd    = min(midCellId+2, ncells_1);

	if ( midCellId == cid ) rowEnd = midCellId + 1; // this row is already partly covered

	const int pstart = valid ? cellsstart[rowStart] : 0;
	const int pend   = valid ? cellsstart[rowEnd] : -1;


	for (int srcId = pstart + idInGroup; __any(srcId < pend); srcId += groupSize)
	{
		const float3 srcCoos = (srcId < pend) ? readCoosFromAll4(xyzouvwo, srcId) : make_float3(-10000.0f);

		for (int dstBase=dstStart; dstBase<dstEnd; dstBase+=nDstPerIter)
		{
#pragma unroll nDstPerIter
			for (int dstId=dstBase; dstId<dstBase+nDstPerIter; dstId++)
			{
				bool interacting =  dstId < dstEnd && distance2(srcCoos, readCoosFromAll4(xyzouvwo, dstId)) < 1.00f;
				if (dstStart <= srcId && srcId < dstEnd && dstId <= srcId) interacting = false;

				const int myentry = warpAggrFetchAdd(nentries, interacting, tid);

				if (interacting) inters[myentry] = pack2int(dstId-dstStart, srcId);
			}

			while (nentries >= warpSize)
			{
				const int2 pids = upackFromint( inters[nentries - warpSize + tid], dstStart );
				execInteractions(warpSize, pids, xyzouvwo, axayaz, adpd, gammadpd, sigmadpd, seed);
				nentries -= warpSize;
			}
		}
	}

	if (tid < nentries)
		execInteractions(nentries, upackFromint(inters[tid], dstStart), xyzouvwo, axayaz, adpd, gammadpd, sigmadpd, seed);
}

__global__ void make_texture_dummy( const float4 * xyzouvwo, float4 * xyzo, float4 * uvwo, const uint n )
{
	const uint pid =  (blockIdx.x * blockDim.x + threadIdx.x);
	if (pid >= n) return;

	xyzo[pid] = xyzouvwo[2*pid+0];
	uvwo[pid] = xyzouvwo[2*pid+1];
}

template<typename T>
struct SimpleDeviceBuffer
{
	int capacity, size;

	T * data;

	SimpleDeviceBuffer(int n = 0): capacity(0), size(0), data(NULL) { resize(n);}

	~SimpleDeviceBuffer()
	{
		if (data != NULL)
			cudaFree(data);

		data = NULL;
	}

	void dispose()
	{
		if (data != NULL)
			cudaFree(data);

		data = NULL;
	}

	void resize(const int n)
	{
		assert(n >= 0);

		size = n;

		if (capacity >= n)
			return;

		if (data != NULL)
			cudaFree(data);

		const int conservative_estimate = (int)ceil(1.1 * n);
		capacity = 128 * ((conservative_estimate + 129) / 128);

		cudaMalloc(&data, sizeof(T) * capacity);

#ifndef NDEBUG
		cudaMemset(data, 0, sizeof(T) * capacity);
#endif
	}
};



void forces_dpd_cuda_nohost( const float * const xyzuvw, const float4 * const xyzouvwo, const ushort4 * const xyzo_half, float * const axayaz,  const int np,
		const int * const cellsstart, const int * const cellscount,
		const float rc,
		const float XL, const float YL, const float ZL,
		const float adpd,
		const float gammadpd,
		const float sigmadpd,
		const float invsqrtdt,
		const float seed, cudaStream_t stream )
{

	static SimpleDeviceBuffer<float4> xyzo, uvwo;

	xyzo.resize(np);
	uvwo.resize(np);

	const int ndens = 4;
	const int nx = round(XL / rc);
	const int ny = round(YL / rc);
	const int nz = round(ZL / rc);

	dim3 nblocks(nx, ny/bDimY, nz/bDimZ);
	dim3 nthreads(32, bDimY, bDimZ);

	cudaFuncSetCacheConfig( smth, cudaFuncCachePreferEqual );

	cudaMemsetAsync( axayaz, 0, sizeof( float )* np * 3, stream );
	make_texture_dummy<<< (np + 1023) / 1024, 1024, 0, stream >>>(xyzouvwo, xyzo.data, uvwo.data, np);
	smth<<< nblocks, nthreads, 0, stream >>>((float2*)xyzuvw, xyzo.data, xyzouvwo, axayaz, cellsstart, nx, ny, nz, nx*ny*nz+1, adpd, gammadpd, sigmadpd*invsqrtdt, seed);
}




