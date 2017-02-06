#pragma once

#include <cuda.h>
#include <cstdint>

#include <core/containers.h>
#include <core/logger.h>
#include <core/helper_math.h>

class CellListInfo
{
public:
	int3 ncells;
	int  totcells;
	float3 domainStart, length;
	float rc, invrc;

	CellListInfo(float _rc, float3 domainStart, float3 length);

// ==========================================================================================================================================
// Common cell functions
// ==========================================================================================================================================
	__device__ __host__ __forceinline__ int encode(int ix, int iy, int iz) const
	{
		return (iz*ncells.y + iy)*ncells.x + ix;
	}

	__device__ __host__ __forceinline__ void decode(int cid, int& ix, int& iy, int& iz) const
	{
		ix = cid % ncells.x;
		iy = (cid / ncells.x) % ncells.y;
		iz = cid / (ncells.x * ncells.y);
	}

	__device__ __host__ __forceinline__ int encodeStartSize(int start, uint8_t size) const
	{
		return start + (size << 26);
	}

	__device__ __host__ __forceinline__ int2 decodeStartSize(int code) const
	{
		return make_int2(code & ((1<<26) - 1), code >> 26);
	}

	template<int dim, bool Clamp = true>
	__device__ __host__ __forceinline__ int getCellIdAlongAxis(const float x) const
	{
		float start;
		int cells;
		if (dim == 0) { start = domainStart.x; cells = ncells.x; }
		if (dim == 1) { start = domainStart.y; cells = ncells.y; }
		if (dim == 2) { start = domainStart.z; cells = ncells.z; }

		const float v = floor(invrc * (x - start));

		if (Clamp)
			return min(cells - 1, max(0, (int)v));
		else
			return v;
	}

	template<bool Clamp = true, typename T>
	__device__ __host__ __forceinline__ int getCellId(const T coo) const
	{
		const int ix = getCellIdAlongAxis<0, Clamp>(coo.x);
		const int iy = getCellIdAlongAxis<1, Clamp>(coo.y);
		const int iz = getCellIdAlongAxis<2, Clamp>(coo.z);

		if (!Clamp)
		{
			if (ix < 0 || ix >= ncells.x  ||  iy < 0 || iy >= ncells.y  ||  iz < 0 || iz >= ncells.z)
				return -1;
		}

		return encode(ix, iy, iz);
	}
};

class CellList : public CellListInfo
{
public:
	ParticleVector* pv;

	DeviceBuffer<int> cellsStart;
	DeviceBuffer<uint8_t> cellsSize;

	CellList(ParticleVector* pv, float rc, float3 domainStart, float3 length);
	CellList(ParticleVector* pv, int3 resolution, float3 domainStart, float3 length);

	CellListInfo cellInfo()
	{
		return *((CellListInfo*)this);
	}
	void setStream(cudaStream_t stream)
	{
		cellsSize.pushStream(stream);
		cellsStart.pushStream(stream);
	}
	void build(cudaStream_t stream);
};
