#pragma once

#include <cuda.h>
#include <cstdint>

#include <core/datatypes.h>
#include <core/logger.h>
#include <core/helper_math.h>

class ParticleVector;

class CellListInfo
{
public:
	int3 ncells;
	int  totcells;
	float3 domainSize;
	float3 h, invh;
	float rc;

	// Cell-list can bear maximum 2^bp particles,
	// with no more than 2^(32-bp) particles per cell
	const int blendingPower = 24;

	CellListInfo(float3 h, float3 domainSize);
	CellListInfo(float rc, float3 domainSize);

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

	__device__ __host__ __forceinline__ int encode(int3 cid3) const
	{
		return (cid3.z*ncells.y + cid3.y)*ncells.x + cid3.x;
	}

	__device__ __host__ __forceinline__ int3 decode(int cid) const
	{
		return make_int3(
				cid % ncells.x,
				(cid / ncells.x) % ncells.y,
				cid / (ncells.x * ncells.y)
		);
	}

	__device__ __host__ __forceinline__ int encodeStartSize(int start, uint8_t size) const
	{
		return start + (size << blendingPower);
	}

	__device__ __host__ __forceinline__ int2 decodeStartSize(uint code) const
	{
		return make_int2(code & ((1<<blendingPower) - 1), code >> blendingPower);
	}

	template<bool Clamp = true>
	__device__ __host__ __forceinline__ int3 getCellIdAlongAxis(const float3 x) const
	{
		const int3 v = make_int3( floorf(invh * (x + 0.5f*domainSize)) );

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
private:
	DeviceBuffer<uint8_t> cellsSize;
	cudaStream_t stream;
	int changedStamp = -1;

	PinnedBuffer<Particle> _coosvels;
	DeviceBuffer<Force>    _forces;

	bool primary = false;

public:
	ParticleVector* pv;

	DeviceBuffer<uint> cellsStartSize;
	DeviceBuffer<int> order;

	PinnedBuffer<Particle> *coosvels;
	DeviceBuffer<Force>    *forces;

	CellList(ParticleVector* pv, float rc, float3 domainSize);
	CellList(ParticleVector* pv, int3 resolution, float3 domainSize);

	CellListInfo cellInfo()
	{
		return *((CellListInfo*)this);
	}

	void setStream(cudaStream_t stream)
	{
		this->stream = stream;

		cellsSize.popStream();
		cellsSize.pushStream(stream);

		cellsStartSize.popStream();
		cellsStartSize.pushStream(stream);

		order.popStream();
		order.pushStream(stream);
	}

	bool isPrimary()   { return primary; }
	void makePrimary() { primary = true; }
	void build();
	void addForces();
};
