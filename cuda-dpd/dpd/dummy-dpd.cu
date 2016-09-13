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

__device__ __forceinline__ int warpAggrFetchAdd(int val, bool pred, int tid)
{
	int res;
	const int mask = ballot(pred); // i-th bit is if lane i is interacting
	const int leader = __ffs(mask) - 1;  // 0-th thread is not always active

	if (tid == leader)
	{
		res = val;
		val += __popc(mask); // n of non-zero bits
	}

	res = __shfl(res, leader);
	return res + __popc( mask & ((1 << tid) - 1) );
}

__device__ __forceinline__ void execInteractions(const int nentries, const int2* inters, const float4* xyzouvwo, float* axayaz, int tid, int dstStart, int dstEnd)
{
	const int2 pids = inters[tid];
	if (pids.x == pids.y || tid >= nentries) return;
	float3 frc = _dpd_interaction(xyzouvwo, pids.x, pids.y);

	float* dest = axayaz + pids.x*3;
	atomicAdd(dest,     frc.x);
	atomicAdd(dest + 1, frc.y);
	atomicAdd(dest + 2, frc.z);

	// If src is in the same cell as dst,
	// interaction will be computed twice
	// no need for atomics in this case
	if (pids.y < dstStart || pids.y >= dstEnd)
	{
		dest = axayaz + pids.y*3;
		atomicAdd(dest,     -frc.x);
		atomicAdd(dest + 1, -frc.y);
		atomicAdd(dest + 2, -frc.z);
	}
}

//__launch_bounds__(128, 14)
__global__ void smth(const float4 * __restrict__ xyzouvwo, const ushort4 *  half_xyzo, float* axayaz, const int * __restrict__ cellsstart, const int * __restrict__ cellscount,
		int nCellsX, int nCellsY, int nCellsZ, int ndens, int np, int ncells)
{
	__shared__ int  nentries[4];
	__shared__ int2 inters[4][32];

	const int tid = threadIdx.x;
	const int wid = threadIdx.z * blockDim.z + threadIdx.y;

	const int cellX0 = blockIdx.x;
	const int cellY0 = blockIdx.y * blockDim.y + threadIdx.y;
	const int cellZ0 = blockIdx.z * blockDim.z + threadIdx.z;

	const int cid = (cellZ0*nCellsY + cellY0)*nCellsX + cellX0;

	const int dstStart = cellsstart[cid];
	const int dstEnd   = cellsstart[cid+1];

	nentries[wid] = 0;

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
		const int cellY = cellY0 + ((groupId > 2) ? groupId - 4 : groupId - 1);
		const int cellZ = cellZ0 + ((groupId > 2) ? 0 : 1);

		bool valid = (cellY >= 0 && cellY < nCellsY && cellZ >= 0 && cellZ < nCellsZ && groupId < 5);

		const int midCellId = (cellZ*nCellsY + cellY)*nCellsX + cellX0;
		int rowStart  = max(midCellId-1, 0);//(cellX0 == 0)         ? midCellId     : midCellId - 1;
		int rowEnd    = min(midCellId+2, ncells+1);
		if (midCellId == cid) rowEnd = midCellId + 1; // this row is already partly covered

		const int pstart = valid ? cellsstart[rowStart] : 0;
		const int pend   = valid ? cellsstart[rowEnd] : -1;

		//#pragma unroll 3
		for (int srcId = pstart + idInGroup; __any(srcId < pend); srcId += groupSize)
		{
			if (srcId < pend)
			{
				const float3 srcCoos = readCoos(half_xyzo, srcId);

#pragma unroll
				for (int i=0; i<nDstPerIter; i++)
				{
					bool interacting = distance2(srcCoos, dstCoos[i]) < 1.02f && curDst+i < dstEnd;
					const int myentry = warpAggrFetchAdd(nentries[wid], interacting, tid);

					if (interacting)
					{
						inters[wid][myentry] = make_int2(curDst+i, srcId);
						//if (myentry > 80) printf("lane %d, nentry: %d\n", tid, myentry);
					}
				}
			}

			if (nentries[wid] >= warpSize)
			{
				execInteractions(nentries[wid], inters[wid] + nentries[wid] - warpSize, xyzouvwo, axayaz, tid, dstStart, dstEnd);
				nentries[wid] -= warpSize;
			}
		}
	}

	if (nentries[wid])
		execInteractions(nentries[wid], inters[wid], xyzouvwo, axayaz, tid, dstStart, dstEnd);
}

__device__ __forceinline__ void loadRow(const ushort4 * __restrict__ half_xyzo, int n, float3* shmem, int tid)
{
	for (int i=tid; i<n; i+=warpSize)
	{
		ushort4 tmp = half_xyzo[i];
		shmem[i] = make_float3(__half2float(tmp.x), __half2float(tmp.y), __half2float(tmp.z));
	}
}

const static __device__ float2 id2cooYZ[] = { {0,0}, {1,0}, {0,1}, {1,1}, {2,1}, {0,2}, {1,2}, {2,2}, {0,3}, {1,3}, {2,3} };

__global__ void smth2(const float4 * __restrict__ xyzouvwo, const ushort4 * __restrict__ half_xyzo, float* axayaz, const int * __restrict__ cellsstart, const int * __restrict__ cellscount,
		float nCellsX, float nCellsY, float nCellsZ, int ndens, int np, int ncells)
{
	__shared__ float3 cache[11][16];

	const int tid = threadIdx.x;
	const int wid = threadIdx.y;

	const float cellX0 = blockIdx.x;
	const float cellY0 = blockIdx.y*2.0f;
	const float cellZ0 = blockIdx.z*2.0f;

	//const int cid = (cellZ0*nCellsY + cellY0)*nCellsX + cellX0;
	//const int blockBaseCellid = (cellZ0*nCellsY + cellY0)*nCellsX + cellX0;

//	const int dstStart = cellsstart[cid];
//	const int dstEnd   = cellsstart[cid+1];

#pragma unroll 3
	for (int row = wid; row < 11; row+=4)
	{
		float cellY = id2cooYZ[row].x + cellY0;
		float cellZ = id2cooYZ[row].y + cellZ0;

		if (cellY >= 0.0f && cellY < nCellsY && cellZ >= 0.0f && cellZ < nCellsZ)
		{
			float midCellId = (cellZ*nCellsY + cellY)*nCellsX + cellX0;

			int rowStart  = max((int)round(midCellId-1.0f), 0);
			int rowEnd    = min((int)round(midCellId+2.0f), ncells+1);

			const int pstart = cellsstart[rowStart];
			const int pend   = cellsstart[rowEnd];

			//printf("start %d end %d,  id %d\n", pstart, pend, (wid << 2) + cellY);
			loadRow(half_xyzo + pstart, (pend - pstart), cache[row], tid);
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
	dim3 nthreads(32, 4, 1);
	const int shmemSize = (nthreads.x * nthreads.y * nthreads.z / 32) * ndens * ndens*3 * 2*2 * sizeof(int) + 4*sizeof(int);

	cudaFuncSetCacheConfig( smth, cudaFuncCachePreferNone );

	cudaMemsetAsync( axayaz, 0, sizeof( float )* np * 3, stream );
	smth2<<< nblocks, nthreads >>>(xyzouvwo, xyzo_half, axayaz, cellsstart, cellscount, nx, ny, nz, ndens, np, nx*ny*nz);
	cudaPeekAtLastError();
}




