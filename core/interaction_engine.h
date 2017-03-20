#include "helper_math.h"
#include <cassert>
#include <type_traits>

#include "celllist.h"
#include "non_cached_rw.h"

__device__ __forceinline__ float4 readCoosFromAll4(const float4* __restrict__ coosvels, int pid)
{
	return coosvels[2*pid];
}

__device__ __forceinline__ float4 readVelsFromAll4(const float4* __restrict__ coosvels, int pid)
{
	return coosvels[2*pid+1];
}

__device__ __forceinline__ void readAll4(const float4* __restrict__ coosvels, int pid, float4& coo, float4& vel)
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
		CellListInfo cinfo,  const uint* __restrict__ cellsStart, int np, Interaction interaction)
{
	const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
	if (dstId >= np) return;

	float4 dstCoo, dstVel;
	float3 dstFrc = make_float3(0.0f);
	readAll4(coosvels, dstId, dstCoo, dstVel);

	const int3 cell0 = cinfo.getCellIdAlongAxis(make_float3(dstCoo));

	for (int cellZ = cell0.z-1; cellZ <= cell0.z+1; cellZ++)
		for (int cellY = cell0.y-1; cellY <= cell0.y; cellY++)
			{
				if ( !(cellY >= 0 && cellY < cinfo.ncells.y && cellZ >= 0 && cellZ < cinfo.ncells.z) ) continue;
				if (cellY == cell0.y && cellZ > cell0.z) continue;

				const int midCellId = cinfo.encode(cell0.x, cellY, cellZ);
				int rowStart  = max(midCellId-1, 0);
				int rowEnd    = min(midCellId+2, cinfo.totcells+1);

				if ( cellY == cell0.y && cellZ == cell0.z ) rowEnd = midCellId + 1; // this row is already partly covered

				const int2 pstart = cinfo.decodeStartSize(cellsStart[rowStart]);
				const int2 pend   = cinfo.decodeStartSize(cellsStart[rowEnd]);

				const int2 start_size = make_int2(pstart.x, pend.x - pstart.x);


#pragma unroll 2
				for (int srcId = start_size.x; srcId < start_size.x + start_size.y; srcId ++)
				{
					const float4 srcCoo = readCoosFromAll4(coosvels, srcId);

					bool interacting = distance2(srcCoo, dstCoo) < 1.00f;
					if (dstId <= srcId && cellY == cell0.y && cellZ == cell0.z) interacting = false;

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
		CellListInfo cinfo,  const uint* __restrict__ cellsStart,
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

	const int3 cell0 = cinfo.getCellIdAlongAxis<false>(make_float3(dstCoo));

	for (int cellZ = max(cell0.z-1, 0); cellZ <= min(cell0.z+1, cinfo.ncells.z-1); cellZ++)
		for (int cellY = max(cell0.y-1, 0); cellY <= min(cell0.y+1, cinfo.ncells.y-1); cellY++)
#if 1
		{
						const int midCellId = cinfo.encode(cell0.x, cellY, cellZ);
						int rowStart  = max(midCellId-1, 0);
						int rowEnd    = min(midCellId+2, cinfo.totcells+1);

						const int2 pstart = cinfo.decodeStartSize(cellsStart[rowStart]);
						const int2 pend   = cinfo.decodeStartSize(cellsStart[rowEnd]);

						const int2 start_size = make_int2(pstart.x, pend.x - pstart.x);
#else
			for (int cellX = max(cell0.x-1, 0); cellX <= min(cell0.x+1, cinfo.ncells.x-1); cellX++)
			{
				const int2 start_size = cinfo.decodeStartSize(cellsStart[cinfo.encode(cellX, cellY, cellZ)]);
#endif

#pragma unroll 2
				for (int srcId = start_size.x; srcId < start_size.x + start_size.y; srcId ++)
				{
					const float4 srcCoo = readCoosFromAll4(srcData, srcId);

					bool interacting = distance2(srcCoo, dstCoo) < 1.00f;

					if (interacting)
					{
						const float4 srcVel = readVelsFromAll4(srcData, srcId);

						float3 frc = interaction(dstCoo, dstVel, __float_as_int(dstCoo.w),
												 srcCoo, srcVel, __float_as_int(srcCoo.w));

						if (NeedDstAcc)
							dstFrc += frc;

						if (NeedSrcAcc)
						{
							float* dest = srcFrcs + srcId*4;

							atomicAdd(dest,     -frc.x);
							atomicAdd(dest + 1, -frc.y);
							atomicAdd(dest + 2, -frc.z);
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



template<bool NeedDstAcc, bool NeedSrcAcc, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions2(
		const float4 * __restrict__ dstData, float* dstFrcs,
		const float4 * __restrict__ srcData, float* srcFrcs,
		CellListInfo cinfo,  const uint* __restrict__ cellsStart,
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

	const int3 cell0 = cinfo.getCellIdAlongAxis<false>(make_float3(dstCoo));

	for (int cellZ = max(cell0.z-1, 0); cellZ <= min(cell0.z+1, cinfo.ncells.z-1); cellZ++)
		for (int cellY = max(cell0.y-1, 0); cellY <= min(cell0.y+1, cinfo.ncells.y-1); cellY++)
#if 0
		{
						const int midCellId = cinfo.encode(cell0.x, cellY, cellZ);
						int rowStart  = max(midCellId-1, 0);
						int rowEnd    = min(midCellId+2, cinfo.totcells+1);

						const int2 pstart = cinfo.decodeStartSize(cellsStart[rowStart]);
						const int2 pend   = cinfo.decodeStartSize(cellsStart[rowEnd]);

						const int2 start_size = make_int2(pstart.x, pend.x - pstart.x);
#else
			for (int cellX = max(cell0.x-1, 0); cellX <= min(cell0.x+1, cinfo.ncells.x-1); cellX++)
			{
				const int2 start_size = cinfo.decodeStartSize(cellsStart[cinfo.encode(cellX, cellY, cellZ)]);
#endif

#pragma unroll 2
				for (int srcId = start_size.x; srcId < start_size.x + start_size.y; srcId ++)
				{
					const float4 srcCoo = readCoosFromAll4(srcData, srcId);

					bool interacting = distance2(srcCoo, dstCoo) < 1.00f;

					if (interacting)
					{
						const float4 srcVel = readVelsFromAll4(srcData, srcId);

						float3 frc = interaction(dstCoo, dstVel, __float_as_int(dstCoo.w),
												 srcCoo, srcVel, __float_as_int(srcCoo.w));

						if (NeedDstAcc)
							dstFrc += frc;

						if (NeedSrcAcc)
						{
							float* dest = srcFrcs + srcId*4;

							atomicAdd(dest,     -frc.x);
							atomicAdd(dest + 1, -frc.y);
							atomicAdd(dest + 2, -frc.z);
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
