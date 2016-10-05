#pragma once

#include <cuda.h>
#include "helper_math.h"
#include <cstdint>

void buildCellList(float4* in_xyzouvwo, const int n, const float3 domainStart, const int3 ncells, const float invrc,
				   float4* out_xyzouvwo, uint8_t* cellsSize, int* cellsStart, cudaStream_t stream);

void buildCellListWithPrecomputedSizes(float4* in_xyzouvwo, const int n, const float3 domainStart, const int3 ncells, const float invrc,
				   float4* out_xyzouvwo, uint8_t* cellsSize, int* cellsStart, cudaStream_t stream);


template<bool Clamp = true>
__device__ __host__ __forceinline__ int getCellIdAlongAxis(const float x, const float start, const int ncells, const float invrc)
{
	const float v = invrc * (x - start);
	const float robustV = min(min(floor(v), floor(v - 1.0e-6f)), floor(v + 1.0e-6f));

	if (Clamp)
		return min(ncells - 1, max(0, (int)robustV));
	else
		return robustV;
}

template<typename T, bool Clamp = true>
__device__ __host__ __forceinline__ int getCellId(const T coo, const float3 domainStart, const int3 ncells, const float invrc)
{
	const int ix = getCellIdAlongAxis<Clamp>(coo.x, domainStart.x, ncells.x, 1.0f);
	const int iy = getCellIdAlongAxis<Clamp>(coo.y, domainStart.y, ncells.y, 1.0f);
	const int iz = getCellIdAlongAxis<Clamp>(coo.z, domainStart.z, ncells.z, 1.0f);

	return (iz*ncells.y + iy)*ncells.x + ix;
}
