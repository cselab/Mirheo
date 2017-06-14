#include <cassert>
#include <type_traits>

#include <core/helper_math.h>
#include <core/celllist.h>
#include <core/cuda_common.h>


template<typename Ta, typename Tb>
__device__ __forceinline__ float distance2(const Ta a, const Tb b)
{
	auto sqr = [] (float x) { return x*x; };
	return sqr(a.x - b.x) + sqr(a.y - b.y) + sqr(a.z - b.z);
}

__device__ __forceinline__ float3 f4tof3(float4 v)
{
	return make_float3(v.x, v.y, v.z);
}

__device__ __forceinline__ void atomicAdd(float* dest, float3 v)
{
	atomicAdd(dest,     v.x);
	atomicAdd(dest + 1, v.y);
	atomicAdd(dest + 2, v.z);
}

template<typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeSelfInteractions(
		const int np, const float4 * __restrict__ coosvels, float* forces,
		CellListInfo cinfo, const uint* __restrict__ cellsStartSize,
		const float rc2, Interaction interaction)
{
	const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
	if (dstId >= np) return;

	const float4 dstCoo = coosvels[2*dstId];
	const float4 dstVel = coosvels[2*dstId+1];
	const Particle dstP(dstCoo, dstVel);
	float3 dstFrc = make_float3(0.0f);

	const int3 cell0 = cinfo.getCellIdAlongAxis(dstP.r);

	for (int cellZ = cell0.z-1; cellZ <= cell0.z+1; cellZ++)
		for (int cellY = cell0.y-1; cellY <= cell0.y; cellY++)
			{
				if ( !(cellY >= 0 && cellY < cinfo.ncells.y && cellZ >= 0 && cellZ < cinfo.ncells.z) ) continue;
				if (cellY == cell0.y && cellZ > cell0.z) continue;

				const int midCellId = cinfo.encode(cell0.x, cellY, cellZ);
				int rowStart  = max(midCellId-1, 0);
				int rowEnd    = min(midCellId+2, cinfo.totcells+1);

				if ( cellY == cell0.y && cellZ == cell0.z ) rowEnd = midCellId + 1; // this row is already partly covered

				const int2 pstart = cinfo.decodeStartSize(cellsStartSize[rowStart]);
				const int2 pend   = cinfo.decodeStartSize(cellsStartSize[rowEnd]);

				const int2 start_size = make_int2(pstart.x, pend.x - pstart.x);


#pragma unroll 2
				for (int srcId = start_size.x; srcId < start_size.x + start_size.y; srcId ++)
				{
					const float4 srcCoo = coosvels[2*srcId];

					bool interacting = distance2(srcCoo, dstP.r) < rc2;
					if (dstId <= srcId && cellY == cell0.y && cellZ == cell0.z) interacting = false;

					if (interacting)
					{
						const float4 srcVel = coosvels[2*srcId+1];
						const Particle srcP(srcCoo, srcVel);

						float3 frc = interaction(dstP, srcP);

						dstFrc += frc;
						if (dot(frc, frc) > 1e-6f)
							atomicAdd(forces + srcId*4, -frc);
					}
				}
			}

	atomicAdd(forces + dstId*4, dstFrc);
}


/**
 * variant == true  better for dense shit,
 * variant == false better for halo and one-sided
 */
template<bool NeedDstAcc, bool NeedSrcAcc, bool Variant, typename Interaction>
__launch_bounds__(128, 16)
__global__ void computeExternalInteractions(
		const int ndst,
		const float4 * __restrict__ dstData, float* dstFrcs,
		const float4 * __restrict__ srcData, float* srcFrcs,
		CellListInfo cinfo,  const uint* __restrict__ cellsStartSize,
		const float rc2, Interaction interaction)
{
	static_assert(NeedDstAcc || NeedSrcAcc, "External interactions should return at least some accelerations");

	const int dstId = blockIdx.x*blockDim.x + threadIdx.x;
	if (dstId >= ndst) return;

	const Particle dstP(readNoCache(dstData+2*dstId), readNoCache(dstData+2*dstId+1));
	float3 dstFrc = make_float3(0.0f);

	const int3 cell0 = cinfo.getCellIdAlongAxis<false>(dstP.r);

	auto computeCell = [&] (int2 start_size) {
#pragma unroll 2
				for (int srcId = start_size.x; srcId < start_size.x + start_size.y; srcId ++)
				{
					const float4 srcCoo = srcData[2*srcId];

					bool interacting = distance2(srcCoo, dstP.r) < rc2;

					if (interacting)
					{
						const float4 srcVel = srcData[2*srcId+1];
						const Particle srcP(srcCoo, srcVel);

						float3 frc = interaction(dstP, srcP);

						if (NeedDstAcc)
							dstFrc += frc;

						if (NeedSrcAcc)
							if (dot(frc, frc) > 1e-6f)
								atomicAdd(srcFrcs + srcId*4, -frc);
					}
				}
	};

	for (int cellZ = max(cell0.z-1, 0); cellZ <= min(cell0.z+1, cinfo.ncells.z-1); cellZ++)
		for (int cellY = max(cell0.y-1, 0); cellY <= min(cell0.y+1, cinfo.ncells.y-1); cellY++)
			if (Variant)
			{
				const int midCellId = cinfo.encode(cell0.x, cellY, cellZ);
				int rowStart  = max(midCellId-1, 0);
				int rowEnd    = min(midCellId+2, cinfo.totcells+1);

				const int2 pstart = cinfo.decodeStartSize(cellsStartSize[rowStart]);
				const int2 pend   = cinfo.decodeStartSize(cellsStartSize[rowEnd]);

				const int2 start_size = make_int2(pstart.x, pend.x - pstart.x);
				computeCell(start_size);
			}
			else
			{
				for (int cellX = max(cell0.x-1, 0); cellX <= min(cell0.x+1, cinfo.ncells.x-1); cellX++)
				{
					const int2 start_size = cinfo.decodeStartSize(cellsStartSize[cinfo.encode(cellX, cellY, cellZ)]);
					computeCell(start_size);
				}
			}

	if (NeedDstAcc)
		atomicAdd(dstFrcs + dstId*4, dstFrc);
}
