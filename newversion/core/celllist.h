#pragma once

#include <cuda.h>
#include "helper_math.h"
#include <cstdint>

#include "containers.h"

void buildCellList(ParticleVector& pv, cudaStream_t stream);

void buildCellListAndIntegrate(ParticleVector& pv, float dt, cudaStream_t stream);


// ==========================================================================================================================================
// Morton or normal cells
// ==========================================================================================================================================
#define __MORTON__

#ifdef __MORTON__
__device__ __host__ inline int getThirdBits(const int m)
{
	int x = m;
	x = x               & 0x09249249;
	x = (x ^ (x >> 2))  & 0x030c30c3;
	x = (x ^ (x >> 4))  & 0x0300f00f;
	x = (x ^ (x >> 8))  & 0xff0000ff;
	x = (x ^ (x >> 16)) & 0x000003ff;

	return x;
}

__device__ __host__ inline int splitBy3bits(const int a)
{
	int x = a;
	x = x               & 0x000003ff;
	x = (x | (x << 16)) & 0xff0000ff;
	x = (x | (x << 8))  & 0x0300f00f;
	x = (x | (x << 4))  & 0x030c30c3;
	x = (x | (x << 2))  & 0x09249249;

	return x;
}

__device__ __host__ int inline encode(int ix, int iy, int iz, int3 ncells)
{
    return splitBy3bits(ix) | (splitBy3bits(iy) << 1) | (splitBy3bits(iz) << 2);
}

__device__ __host__ int3 inline decode(int code, int3 ncells)
{
    return make_int3(
    getThirdBits(code),
    getThirdBits(code >> 1),
    getThirdBits(code >> 2)
	);
}
#else

__device__ __host__ __forceinline__ float encode(int ix, int iy, int iz, int3 ncells)
{
	return (iz*ncells.y + iy)*ncells.x + ix;
}

#endif

// ==========================================================================================================================================
// Common cell functions
// ==========================================================================================================================================

__device__ __host__ __forceinline__ int encodeStartSize(int start, uint8_t size)
{
	return start + (size << 26);
}

__device__ __host__ __forceinline__ int2 decodeStartSize(int code)
{
	return make_int2(code & ((1<<26) - 1), code >> 26);
}

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

template<bool Clamp = true, typename T>
__device__ __host__ __forceinline__ int getCellId(const T coo, const float3 domainStart, const int3 ncells, const float invrc)
{
	const int ix = getCellIdAlongAxis<Clamp>(coo.x, domainStart.x, ncells.x, 1.0f);
	const int iy = getCellIdAlongAxis<Clamp>(coo.y, domainStart.y, ncells.y, 1.0f);
	const int iz = getCellIdAlongAxis<Clamp>(coo.z, domainStart.z, ncells.z, 1.0f);

	if (!Clamp)
	{
		if (ix < 0 || ix >= ncells.x  ||  iy < 0 || iy >= ncells.y  ||  iz < 0 || iz >= ncells.z)
			return -1;
	}

	return encode(ix, iy, iz, ncells);
}
