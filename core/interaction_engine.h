#include "helper_math.h"
#include <cassert>
#include <type_traits>

#include "celllist.h"
#include "non_cached_rw.h"

__device__ __forceinline__ float4 readCoosFromAll4(const float4* coosvels, int pid)
{
	return coosvels[2*pid];
}

__device__ __forceinline__ float4 readVelsFromAll4(const float4* coosvels, int pid)
{
	return coosvels[2*pid+1];
}

__device__ __forceinline__ void readAll4(const float4* coosvels, int pid, float4& coo, float4& vel)
{
	coo = coosvels[pid*2];
	vel = coosvels[pid*2+1];
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

__device__ __forceinline__ float3 f4tof3(float4 v)
{
	return make_float3(v.x, v.y, v.z);
}

template<typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeSelfInteractions(const float4 * __restrict__ coosvels, float* forces,
		CellListInfo cinfo,  const int* __restrict__ cellsStart, int np, Interaction interaction)
{
	const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
	if (dstId >= np) return;

	float4 dstCoo, dstVel;
	float3 dstFrc = make_float3(0.0f);
	readAll4(coosvels, dstId, dstCoo, dstVel);

	const int cellX0 = cinfo.getCellIdAlongAxis<0>(dstCoo.x);
	const int cellY0 = cinfo.getCellIdAlongAxis<1>(dstCoo.y);
	const int cellZ0 = cinfo.getCellIdAlongAxis<2>(dstCoo.z);

	for (int cellZ = cellZ0-1; cellZ <= cellZ0+1; cellZ++)
		for (int cellY = cellY0-1; cellY <= cellY0; cellY++)
			{
				if ( !(cellY >= 0 && cellY < cinfo.ncells.y && cellZ >= 0 && cellZ < cinfo.ncells.z) ) continue;
				if (cellY == cellY0 && cellZ > cellZ0) continue;

				const int midCellId = cinfo.encode(cellX0, cellY, cellZ);
				int rowStart  = max(midCellId-1, 0);
				int rowEnd    = min(midCellId+2, cinfo.totcells+1);

				if ( cellY == cellY0 && cellZ == cellZ0 ) rowEnd = midCellId + 1; // this row is already partly covered

				const int2 pstart = cinfo.decodeStartSize(cellsStart[rowStart]);
				const int2 pend   = cinfo.decodeStartSize(cellsStart[rowEnd]);

				const int2 start_size = make_int2(pstart.x, pend.x - pstart.x);


#pragma unroll 2
				for (int srcId = start_size.x; srcId < start_size.x + start_size.y; srcId ++)
				{
					const float4 srcCoo = readCoosFromAll4(coosvels, srcId);

					bool interacting = distance2(srcCoo, dstCoo) < 1.00f;
					if (dstId <= srcId && cellY == cellY0 && cellZ == cellZ0) interacting = false;

					if (interacting)
					{
						const float4 srcVel = readVelsFromAll4(coosvels, srcId);

						float3 frc = interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId);

						dstFrc += frc;

						float* dest = forces + srcId*4;
						atomicAdd(dest,     -frc.x);
						atomicAdd(dest + 1, -frc.y);
						atomicAdd(dest + 2, -frc.z);
					}
				}
			}

	float* dest = forces + dstId*4;
	atomicAdd(dest,     dstFrc.x);
	atomicAdd(dest + 1, dstFrc.y);
	atomicAdd(dest + 2, dstFrc.z);
}

template<bool NeedDstAcc, bool NeedSrcAcc, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions(
		const float4 * __restrict__ dstData, float* dstFrcs,
		const float4 * __restrict__ srcData, float* srcFrcs,
		CellListInfo cinfo,  const int* __restrict__ cellsStart,
		int ndst, Interaction interaction)
{
	static_assert(NeedDstAcc || NeedSrcAcc, "External interactions should return at least some accelerations");

	const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
	if (dstId >= ndst) return;

	float4 dstCoo, dstVel;
	float3 dstFrc = make_float3(0.0f);
	//readAll4(dstData, dstId, dstCoo, dstVel);
	dstCoo = readNoCache(dstData+2*dstId);
	dstVel = readNoCache(dstData+2*dstId+1);

	const int cellX0 = cinfo.getCellIdAlongAxis<0, false>(dstCoo.x);
	const int cellY0 = cinfo.getCellIdAlongAxis<1, false>(dstCoo.y);
	const int cellZ0 = cinfo.getCellIdAlongAxis<2, false>(dstCoo.z);

	for (int cellZ = max(cellZ0-1, 0); cellZ <= min(cellZ0+1, cinfo.ncells.z-1); cellZ++)
		for (int cellY = max(cellY0-1, 0); cellY <= min(cellY0+1, cinfo.ncells.y-1); cellY++)
			for (int cellX = max(cellX0-1, 0); cellX <= min(cellX0+1, cinfo.ncells.x-1); cellX++)
			{
				const int2 start_size = cinfo.decodeStartSize(cellsStart[cinfo.encode(cellX, cellY, cellZ)]);

#pragma unroll 2
				for (int srcId = start_size.x; srcId < start_size.x + start_size.y; srcId ++)
				{
					const float4 srcCoo = readCoosFromAll4(srcData, srcId);

					bool interacting = distance2(srcCoo, dstCoo) < 1.00f;

					if (interacting)
					{
						const float4 srcVel = readVelsFromAll4(srcData, srcId);

						float3 frc = interaction(dstCoo, dstVel, dstId, srcCoo, srcVel, srcId);

						if (NeedDstAcc)
							dstFrc += frc;

						if (NeedSrcAcc)
						{
							float* dest = srcFrcs + srcId*4;

							float a = atomicAdd(dest,     -frc.x);
							float b = atomicAdd(dest + 1, -frc.y);
							float c = atomicAdd(dest + 2, -frc.z);
						}
					}
				}
			}

	if (NeedDstAcc)
	{
		float* dest = dstFrcs + dstId*4;
		dest[0] += dstFrc.x;
		dest[1] += dstFrc.y;
		dest[2] += dstFrc.z;
	}
}
