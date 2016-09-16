#include <cstdio>
#include <helper_math.h>
#include <cuda_fp16.h>
#include <cassert>
#include "../tiny-float.h"

#include "cuda-dpd.h"
#include "../dpd-rng.h"

__device__ __forceinline__ float3 readCoosHalf(const ushort4* half_xyzo, int pid)
{
	const ushort4 tmp = half_xyzo[pid];

	return make_float3(__half2float(tmp.x), __half2float(tmp.y), __half2float(tmp.z));
}

__device__ __forceinline__ float3 readCoos4_coos(const float4* xyzo, int pid)
{
	const float4 tmp = xyzo[pid];

	return make_float3(tmp.x, (tmp.y), (tmp.z));
}

__device__ __forceinline__ float3 readCoos4(const float4* xyzouvwo, int pid)
{
	const float4 tmp = xyzouvwo[pid*2];

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

__device__ __forceinline__ float3 warpReduceSum(float3 val)
{
	for (int offset = warpSize/2; offset > 0; offset /= 2)
	{
		val.x += __shfl_down(val.x, offset);
		val.y += __shfl_down(val.y, offset);
		val.z += __shfl_down(val.z, offset);
	}
	return val;
}


//const float dt = 0.0025;
const float kBT = 1.0;
const float gammadpd = 20;
const float sigma = sqrt(2 * gammadpd * kBT);
const float sigmaf = 126.4911064;//sigma / sqrt(dt);
const float aij = 50;

__device__ __forceinline__ float3 _dpd_interaction(const float4* xyzo, const float4* uvwo, const int dstId, const int srcId)
{
	const float3 dstCoo = readCoos4_coos(xyzo, dstId);
	const float3 dstVel = readCoos4_coos(uvwo, dstId);

	const float3 srcCoo = readCoos4_coos(xyzo, srcId);
	const float3 srcVel = readCoos4_coos(uvwo, srcId);

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

	const float myrandnr = 0;//Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

	const float strength = aij * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

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

__device__ __forceinline__ void execInteractions(const int nentries, const int2 pids, const float4* xyzo, const float4* uvwo, float* axayaz, int dstStart, int dstEnd)
{
	//assert(nentries <= 32);

	if (pids.x == pids.y) return;
	float3 frc = _dpd_interaction(xyzo, uvwo, pids.x, pids.y);
	//
	float* dest = axayaz + xscale(pids.x, 3.0f);
	atomicAdd(dest,     frc.x);
	atomicAdd(dest + 1, frc.y);
	atomicAdd(dest + 2, frc.z);

	// If src is in the same cell as dst,
	// interaction will be computed twice
	// no need for atomics in this case
	dest = axayaz + xscale(pids.y, 3.0f);
	atomicAdd(dest,     -frc.x);
	atomicAdd(dest + 1, -frc.y);
	atomicAdd(dest + 2, -frc.z);
}

const int bDimY = 2;
const int bDimZ = 2;
const int nDstPerIter = 4;

__launch_bounds__(32*bDimY*bDimZ, 64 / (bDimY*bDimZ))
__global__ void smth(const float4 * __restrict__ xyzo, const float4 * __restrict__ uvwo, float* axayaz, const int * __restrict__ cellsstart, int nCellsX, int nCellsY, int nCellsZ, int ncells)
{
	const int tid = threadIdx.x;
	const int wid = (threadIdx.z * bDimZ + threadIdx.y);

	volatile __shared__ int pool[ bDimY*bDimZ ][128];
	volatile int* inters = pool[wid];
	int nentries = 0;

	const int cellX0 = (blockIdx.x) ;
	const int cellY0 = xmad(blockIdx.y, (float)bDimY, threadIdx.y);//(blockIdx.y * bDimY) + threadIdx.y;
	const int cellZ0 = xmad(blockIdx.z, (float)bDimZ, threadIdx.z);//(blockIdx.z * bDimZ) + threadIdx.z;

	const int cid = (cellZ0*nCellsY + cellY0)*nCellsX + cellX0;

	const int dstStart = cellsstart[cid];
	const int dstEnd   = cellsstart[cid+1];

	const int groupSize = 6;
	const int groupId = xdiv((uint)tid, 0.16666666667f);//tid / groupSize;
	const int idInGroup = xsub(tid, xscale(groupId, 6.0f));//groupSize * groupId;

	int cellY = cellY0 + (uint)groupId % 3 - 1;
	int cellZ = cellZ0 + (uint)groupId / 3 - 1;

	bool valid = (cellY >= 0 && cellY < nCellsY && cellZ >= 0 && cellZ < nCellsZ && groupId < 5);

	const int midCellId = (cellZ*nCellsY + cellY)*nCellsX + cellX0;
	int rowStart  = max(midCellId-1, 0);//(cellX0 == 0)         ? midCellId     : midCellId - 1;
	int rowEnd    = min(midCellId+2, ncells);
	if ( midCellId == cid ) rowEnd = midCellId + 1; // this row is already partly covered

	const int pstart = cellsstart[rowStart];
	const int pend   = valid ? cellsstart[rowEnd] : -1;

	for (int srcId = pstart + idInGroup; __any(srcId < pend); srcId += groupSize)
	{
		const float3 srcCoos = (srcId < pend) ? readCoos4_coos(xyzo, srcId) : make_float3(-10000.0f);

		for (int dstBase=dstStart; dstBase<dstEnd; dstBase+=nDstPerIter)
		{
#pragma unroll nDstPerIter
			for (int dstId=dstBase; dstId<dstBase+nDstPerIter; dstId++)
			{
				bool interacting = distance2(srcCoos, readCoos4_coos(xyzo, dstId)) < 1.00f && dstId < dstEnd;
				if (dstStart <= srcId && srcId < dstEnd && dstId <= srcId) interacting = false;

				const int myentry = warpAggrFetchAdd(nentries, interacting, tid);
				if (interacting) inters[myentry] = pack2int(dstId-dstStart, srcId);
			}

			while (nentries >= warpSize)
			{
				const int2 pids = upackFromint( inters[nentries - warpSize + tid], dstStart );
				execInteractions(warpSize, pids, xyzo, uvwo, axayaz, dstStart, dstEnd);
				nentries -= warpSize;
			}
		}
	}

	if (tid < nentries)
		execInteractions(nentries, upackFromint(inters[tid], dstStart), xyzo, uvwo, axayaz, dstStart, dstEnd);
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
		const float aij,
		const float gamma,
		const float sigma,
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
	make_texture_dummy<<< (np + 1023) / 1024, 1024 >>>(xyzouvwo, xyzo.data, uvwo.data, np);
	smth<<< nblocks, nthreads >>>(xyzo.data, uvwo.data, axayaz, cellsstart, nx, ny, nz, nx*ny*nz);
}




