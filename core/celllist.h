#pragma once

#include <cuda.h>
#include <cstdint>

#include <core/datatypes.h>
#include <core/containers.h>
#include <core/logger.h>
#include <core/helper_math.h>

class CellListInfo
{
public:
	int3 ncells;
	int  totcells;
	float3 domainStart, length;
	float3 h, invh;
	float rc;

	CellListInfo(float3 h, float3 domainStart, float3 length);
	CellListInfo(float rc, float3 domainStart, float3 length);

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
		return start + (size << 24);
	}

	__device__ __host__ __forceinline__ int2 decodeStartSize(int code) const
	{
		return make_int2(code & ((1<<24) - 1), code >> 24);
	}

	template<bool Clamp = true>
	__device__ __host__ __forceinline__ int3 getCellIdAlongAxis(const float3 x) const
	{
		const int3 v = make_int3( floorf(invh * (x - domainStart)) );

		if (Clamp)
			return min( ncells - 1, max(make_int3(0), v) );
		else
			return v;
	}

	template<bool Clamp = true, typename T>
	__device__ __host__ __forceinline__ int getCellId(const T coo) const
	{
		const int3 id = getCellIdAlongAxis<Clamp>(make_float3(coo));

		if (!Clamp)
		{
			if (id.x < 0 || id.x >= ncells.x  ||  id.y < 0 || id.y >= ncells.y  ||  id.z < 0 || id.z >= ncells.z)
				return -1;
		}

		return encode(id.x, id.y, id.z);
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
