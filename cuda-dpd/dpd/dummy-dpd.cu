#include <cstdio>
#include <helper_math.h>
#include <cuda_fp16.h>

#include "cuda-dpd.h"
#include "../dpd-rng.h"

const int nDstPerIter = 4;

__device__ __forceinline__ float3 readCoos(const ushort4* half_xyzo, int pid)
{
	const ushort4 tmp = half_xyzo[pid];

	return make_float3(__half2float(tmp.x), __half2float(tmp.y), __half2float(tmp.z));
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


const float dt = 0.0025;
const float kBT = 1.0;
const float gammadpd = 20;
const float sigma = sqrt(2 * gammadpd * kBT);
const float sigmaf = 126.4911064;//sigma / sqrt(dt);
const float aij = 50;

__device__ __forceinline__ float3 _dpd_interaction(const float4* __restrict__ xyzouvwo, const int dstId, const int srcId)
{
	const float4 dstCoo = xyzouvwo[dstId*2 + 0];
	const float4 dstVel = xyzouvwo[dstId*2 + 1];

	const float4 srcCoo = xyzouvwo[srcId*2 + 0];
	const float4 srcVel = xyzouvwo[srcId*2 + 1];

	const float _xr = dstCoo.x - srcCoo.x;
	const float _yr = dstCoo.y - srcCoo.y;
	const float _zr = dstCoo.z - srcCoo.z;
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
			xr * (dstVel.x - srcVel.x) +
			yr * (dstVel.y - srcVel.y) +
			zr * (dstVel.z - srcVel.z);

	const float myrandnr = Logistic::mean0var1(1, min(srcId, dstId), max(srcId, dstId));

	const float strength = aij * argwr - (gammadpd * wr * rdotv + sigmaf * myrandnr) * wr;

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

//__launch_bounds__(128, 14)
__global__ void smth(const float4 * __restrict__ xyzouvwo, const ushort4 *  half_xyzo, float* axayaz, const int * __restrict__ cellsstart, const int * __restrict__ cellscount,
		int nCellsX, int nCellsY, int nCellsZ, int ndens, int np, int ncells)
{
	extern __shared__ int shmem[];

	const int tid = threadIdx.x;
	const int wid = threadIdx.z * blockDim.z + threadIdx.y;

	int*  myshmem  = shmem + wid * ndens * ndens*3 * 2*2 + 1;
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
			dstCoos[i] = readCoos(half_xyzo, curDst+i);

		// 5 groups 6 threads each
		// each group takes one row
		const int groupSize = 6;
		const int groupId = tid / groupSize;
		const int idInGroup = tid - groupSize*groupId;
		const int cellY = cellY0 + (groupId > 2) ? groupId - 4 : groupId - 1;
		const int cellZ = cellZ0 + (groupId > 2) ? 0 : 1;

		if (cellY < 0 || cellY >= nCellsY || cellZ < 0 || cellZ >= nCellsZ) break;
		if (groupId > 4) break;

		const int midCellId = (cellZ*nCellsY + cellY)*nCellsX + cellX0;
		int rowStart  = max(midCellId-1, 0);//(cellX0 == 0)         ? midCellId     : midCellId - 1;
		int rowEnd    = min(midCellId+2, ncells+1);
		if (midCellId == cid) rowEnd = midCellId + 1; // this row is already partly covered

		const int pstart = cellsstart[rowStart];
		const int pend   = cellsstart[rowEnd];

#pragma unroll 1
		for (int srcId = pstart + idInGroup; srcId < pend; srcId += groupSize)
		{
			const float3 srcCoos = readCoos(half_xyzo, srcId);

#pragma unroll
			for (int i=0; i<nDstPerIter; i++)
			{
				bool interacting = distance2(srcCoos, dstCoos[i]) < 1.01f && curDst+i < dstEnd;
				const int myentry = warpAggrFetchAdd(nentries, interacting);

				if (interacting)
				{
					const int dstId = curDst+i;
					if (srcId == dstId) continue;
					float3 frc = _dpd_interaction(xyzouvwo, dstId, srcId);

					atomicAdd(axayaz + dstId*3 + 0, frc.x);
					atomicAdd(axayaz + dstId*3 + 1, frc.y);
					atomicAdd(axayaz + dstId*3 + 2, frc.z);

					atomicAdd(axayaz + srcId*3 + 0, -frc.x);
					atomicAdd(axayaz + srcId*3 + 1, -frc.y);
					atomicAdd(axayaz + srcId*3 + 2, -frc.z);


					inters[myentry] = make_int2(curDst+i, frc.x);
					//if (myentry > 80) printf("lane %d, nentry: %d\n", tid, myentry);
				}
			}
		}
	}

	return;

	const int totEntries = *nentries;
//	if (tid == 0)
//		printf("cid %02d (%d %d %d):   %d\n", cid, cellX0, cellY0, cellZ0, totEntries/4);

#pragma unroll 2
	for (int i=tid; i<totEntries; i+=warpSize)
	{
		const int2 pids = inters[i];
		if (pids.x == pids.y) continue;
		float3 frc = _dpd_interaction(xyzouvwo, pids.x, pids.y);

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
	const int shmemSize = (nthreads.x * nthreads.y * nthreads.z / 32) * ndens * ndens*3 * 2*2 * sizeof(int) + 4*sizeof(int);

	cudaFuncSetCacheConfig( smth, cudaFuncCachePreferNone );

	cudaMemsetAsync( axayaz, 0, sizeof( float )* np * 3, stream );
	smth<<< nblocks, nthreads, shmemSize >>>(xyzouvwo, xyzo_half, axayaz, cellsstart, cellscount, nx, ny, nz, ndens, np, nx*ny*nz);
	cudaPeekAtLastError();
}




