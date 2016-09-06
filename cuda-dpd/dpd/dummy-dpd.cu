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

__global__ void smth(const float * __restrict__ xyzuvw, const ushort4 * __restrict__ hxyzo, float* axayaz, const int * const cellsstart, const int * const cellscount,
		int nCellsX, int nCellsY, int nCellsZ, int ndens, int np, int ncells)
{
	extern __shared__ int shmem[];

	const int globId = blockIdx.x * blockDim.x + threadIdx.x;
	const int cid = globId >> 5;
	const int tid = globId & 31;
	const int wid = threadIdx.x >> 5;

	int* table  = shmem + wid * 2*ndens * (2*ndens*4 + 1);

	if (cid >= ncells) return;

	const int cellX0 = cid % nCellsX;
	const int cellY0 = (cid / nCellsX) % nCellsY;
	const int cellZ0 = (cid / (nCellsX*nCellsY)) % nCellsZ;

	const int dstStart = cellsstart[cid];
	const int dstEnd   = cellsstart[cid+1];

	for (int dstShift = 0; dstShift < dstEnd - dstStart; dstShift += nDstPerIter)
	{
		float3 dstCoos[nDstPerIter];

#pragma unroll
		for (int i=0; i<nDstPerIter; i++)
		{
			dstCoos[i] = readCoos(xyzuvw, dstStart + dstShift + i);
			table[(dstShift + i)*(2*ndens*4 + 1)] = 0;
		}

		const int deltaCell = tid / 6;
		const int cellY = cellY0 + deltaCell % 3 - 1;
		const int cellZ = cellZ0 + deltaCell / 3 - 1;
		if (cellY < 0 || cellY == nCellsY || cellZ < 0 || cellZ == nCellsZ) continue;

		int rowStartCell = (cellZ*nCellsY + cellY)*nCellsX + cellX0 - 1;
		const int pstart = cellsstart[max(rowStartCell, 0)];
		const int pend   = cellsstart[min(rowStartCell+3, ncells)];

#pragma unroll 2
		for (int srcId = tid % 6; srcId < pend - pstart; srcId += 6)
		{
			const float3 srcCoos = readCoos(xyzuvw, srcId + pstart);

#pragma unroll
			for (int i=0; i<nDstPerIter; i++)
				if (distance2(srcCoos, dstCoos[i]) < 1.0f)
				{
					int* intList = table + (dstShift + i)*(2*ndens*4 + 1);
					const int myentry = atomicAdd(intList, 1) + 1; // +1 cause first element is size
//					if (cid == 48*48*2 + 48*2 + 10)
//						printf("dst: %d, entry: %d\n", dstShift + i, myentry);

					intList[myentry] = srcId + pstart;
				}
		}
	}


	return;
	const int groupSize = 8;
	const int groupId = tid / groupSize;
	const int idInGroup = tid % groupId;

	// Each group works on one dst
	for (int dst = groupId; dst < dstEnd - dstStart; dst += groupSize)
	{
		// Interaction list for this dst
		const int* intList = table + dst*(2*ndens*4 + 1);
		const int nentires = intList[0];
		float3 acc = make_float3(0, 0, 0);

#pragma unroll 2
		for (int i=idInGroup; i<nentires; i+=groupSize)
		{
			const int srcId = intList[i + 1]; // NOTE +1 !!
			float3 frc = _dpd_interaction((float2*)xyzuvw, dst + dstStart, srcId);

			acc += frc;

			atomicAdd(axayaz + srcId*3 + 0, -frc.x);
			atomicAdd(axayaz + srcId*3 + 1, -frc.y);
			atomicAdd(axayaz + srcId*3 + 2, -frc.z);
		}

		axayaz[(dst + dstStart)*3 + 0] = acc.x;
		axayaz[(dst + dstStart)*3 + 1] = acc.y;
		axayaz[(dst + dstStart)*3 + 2] = acc.z;
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
	const int nthreads = 128;

	// 2*ndens rows of one count + 2*ndens*4 entries
	const int shmemSize = (nthreads / 32) * 2*ndens * (2*ndens*4 + 1) * sizeof(int);

	printf("SHMEM: %d\n", shmemSize);

	smth<<< round(32 * XL*YL*ZL + nthreads-1) / nthreads, nthreads, shmemSize >>>(xyzuvw, xyzo_half, axayaz, cellsstart, cellscount, XL, YL, ZL, ndens, np, XL*YL*ZL);
	cudaPeekAtLastError();
	cudaMemsetAsync( axayaz, 0, sizeof( float )* np * 3, stream ) ;
}




