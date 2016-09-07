#include <cstdio>
#include <helper_math.h>
#include <cuda_fp16.h>

#include "cuda-dpd.h"
#include "../dpd-rng.h"

const int nDstPerIter = 4;

__device__ __forceinline__ float3 readCoos(const float* xyzuvw, int pid)
{
	const float2 tmp1 = *((float2*)xyzuvw + pid*3);
	const float2 tmp2 = *((float2*)xyzuvw + pid*3 + 1);

	return make_float3(tmp1.x, tmp1.y, tmp2.x);
}

__device__ __forceinline__ void readRow(const float2 * xyzuvw, int n, float2* shmem, int tid)
{
	for (int i=tid; i<n; i+=warpSize)
		shmem[i] = xyzuvw[i];
}

__device__ __forceinline__ float sqr(float x)
{
	return x*x;
}
__device__ __forceinline__ float distance2(float3 a, float3 b)
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

__device__ __forceinline__ float3 _dpd_interaction(float2* xyzuvw, const int dstId, const int srcId)
{
	const float2 dtmp0 = xyzuvw[dstId*3];
	const float2 dtmp1 = xyzuvw[dstId*3 + 1];
	const float2 dtmp2 = xyzuvw[dstId*3 + 2];

	const float2 stmp0 = xyzuvw[srcId*3];
	const float2 stmp1 = xyzuvw[srcId*3 + 1];
	const float2 stmp2 = xyzuvw[srcId*3 + 2];

	const float _xr = dtmp0.x - stmp0.x;
	const float _yr = dtmp0.y - stmp0.y;
	const float _zr = dtmp1.x - stmp1.x;
	const float rij2 = _xr * _xr + _yr * _yr + _zr * _zr;
	//assert(rij2 < 1);

	const float invrij = rsqrtf(rij2);
	const float rij = rij2 * invrij;
	const float argwr = 1.0f - rij;
	const float wr = viscosity_function<0>(argwr);

	const float xr = _xr * invrij;
	const float yr = _yr * invrij;
	const float zr = _zr * invrij;

	const float rdotv =
			xr * (dtmp1.y - stmp1.y) +
			yr * (dtmp2.x - stmp2.x) +
			zr * (dtmp2.y - stmp2.y);

	const float myrandnr = Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

	const float strength = 20 * argwr - (45 * wr * rdotv + 50 * myrandnr) * wr;

	return make_float3(strength * xr, strength * yr, strength * zr);
}

__device__ __forceinline__ int warpAggrFetchAdd(int* addr, bool pred)
{
	const int tid = threadIdx.x & 31;
	int res;
	const int mask = ballot(pred); // i-th bit is if lane i is interacting
	const int leader = __ffs(mask) - 1;  // 0-th thread is not always active

	if (tid == leader)
	{
		res = *addr;
		*addr += __popc(mask); // n of non-zero bits
	}

	res = __shfl(res, leader);
	return res + __popc( mask & ((1 << tid) - 1) );
}

__global__ void smth(const float * __restrict__ xyzuvw, const ushort4 * __restrict__ hxyzo, float* axayaz, const int * const cellsstart, const int * const cellscount,
		int nCellsX, int nCellsY, int nCellsZ, int ndens, int np, int ncells)
{
	extern __shared__ int shmem[];

	const int tid = threadIdx.x;
	const int wid = threadIdx.z * blockDim.z + threadIdx.y;

	int*  myshmem  = shmem + wid * ndens * ndens*4 * 2*2*4 + 1;
	int*  nentries = myshmem;
	int2* inters   = (int2*)(myshmem+1);

	const int cellX0 = blockIdx.x;
	const int cellY0 = blockIdx.y * blockDim.y + threadIdx.y;
	const int cellZ0 = blockIdx.z * blockDim.z + threadIdx.z;

	const int cid = (cellZ0*nCellsY + cellY0)*nCellsX + cellX0;

	const int dstStart = cellsstart[cid];
	const int dstEnd   = cellsstart[cid+1];

	*nentries = 0;

	for (int curDst=dstStart; curDst < dstEnd; curDst+=nDstPerIter)
	{
		float3 dstCoos[nDstPerIter];

#pragma unroll
		for (int i=0; i<nDstPerIter; i++)
		{
			dstCoos[i] = readCoos(xyzuvw, curDst+i);
		}

		// 5 groups 6 threads each
		// each group takes one row
		const int groupSize = 6;
		const int groupId = tid / groupSize;
		const int idInGroup = tid % groupSize;
		const int cellY = cellY0 + groupId % 3 - 1;
		const int cellZ = cellZ0 + groupId / 3 - 1;

		if (cellY < 0 || cellY >= nCellsY || cellZ < 0 || cellZ >= nCellsZ) continue;

		const int midCellId = (cellZ*nCellsY + cellY)*nCellsX + cellX0;
		int rowStart  = (cellX0 == 0)         ? midCellId     : midCellId - 1;
		int rowEnd    = (cellX0 == nCellsX-1) ? midCellId + 1 : midCellId + 2;
		if (midCellId == cid) rowEnd = midCellId + 1; // this row is already partly covered

		const int pstart = cellsstart[rowStart];
		const int pend   = cellsstart[rowEnd];

#pragma unroll 2
		for (int srcId = pstart + idInGroup; srcId < pend; srcId += groupSize)
		{
			const float3 srcCoos = readCoos(xyzuvw, srcId);

#pragma unroll
			for (int i=0; i<nDstPerIter; i++)
			{
				const bool interacting = distance2(srcCoos, dstCoos[i]) < 1.0f;
				const int myentry = warpAggrFetchAdd(nentries, interacting);

				if (interacting)
				{
					inters[myentry] = make_int2(curDst+i, srcId);
					//if (cid == 60) printf("lane %d, nentry: %d\n", tid, myentry);
				}
			}
		}
	}

	const int totEntries = *nentries;
//	if (tid == 0)
//		printf("cid %02d (%d %d %d):   %d\n", cid, cellX0, cellY0, cellZ0, totEntries/4);

#pragma unroll 2
	for (int i=tid; i<totEntries; i+=warpSize)
	{
		const int2 pids = inters[i];
		if (pids.x == pids.y) continue;
		float3 frc = _dpd_interaction((float2*)xyzuvw, pids.x, pids.y);

		atomicAdd(axayaz + pids.x*3 + 0, frc.x);
		atomicAdd(axayaz + pids.x*3 + 1, frc.y);
		atomicAdd(axayaz + pids.x*3 + 2, frc.z);

		// If src is in the same cell as dst,
		// interaction will be computed twice
		// no need for atomics in this case
		if (pids.y < dstStart || pids.y >= dstEnd)
		{
			atomicAdd(axayaz + pids.y*3 + 0, -frc.x);
			atomicAdd(axayaz + pids.y*3 + 1, -frc.y);
			atomicAdd(axayaz + pids.y*3 + 2, -frc.z);
		}
	}
}



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

	const int ndens = 4;
	const int nx = round(XL / rc);
	const int ny = round(YL / rc);
	const int nz = round(ZL / rc);

	dim3 nblocks(nx, ny/2, nz/2);
	dim3 nthreads(32, 2, 2);
	const int shmemSize = (nthreads.x * nthreads.y * nthreads.z / 32) * ndens * ndens*4 * 2*2 * sizeof(int)*4 + 4*sizeof(int);

	cudaFuncSetCacheConfig( smth, cudaFuncCachePreferNone );

	cudaMemsetAsync( axayaz, 0, sizeof( float )* np * 3, stream );
	smth<<< nblocks, nthreads, shmemSize >>>(xyzuvw, xyzo_half, axayaz, cellsstart, cellscount, nx, ny, nz, ndens, np, nx*ny*nz);
	cudaPeekAtLastError();
}




