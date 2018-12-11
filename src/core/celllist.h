#pragma once

#include <cstdint>

#include <core/datatypes.h>
#include <core/containers.h>
#include <core/logger.h>
#include <core/utils/cuda_common.h>

class ParticleVector;
struct PVview;

enum class CellListsProjection
{
    Clamp, NoClamp
};


class CellListInfo
{
public:
    int3 ncells;
    int  totcells;
    float3 localDomainSize;
    float3 h, invh;
    float rc;

    int *cellSizes, *cellStarts, *order;
    float4 *particles, *forces;

    CellListInfo(float3 h, float3 localDomainSize);
    CellListInfo(float rc, float3 localDomainSize);

#ifdef __CUDACC__
// ==========================================================================================================================================
// Common cell functions
// ==========================================================================================================================================
    __device__ __host__ inline int encode(int ix, int iy, int iz) const
    {
        return (iz*ncells.y + iy)*ncells.x + ix;
    }

    __device__ __host__ inline void decode(int cid, int& ix, int& iy, int& iz) const
    {
        ix = cid % ncells.x;
        iy = (cid / ncells.x) % ncells.y;
        iz = cid / (ncells.x * ncells.y);
    }

    __device__ __host__ inline int encode(int3 cid3) const
    {
        return encode(cid3.x, cid3.y, cid3.z);
    }

    __device__ __host__ inline int3 decode(int cid) const
    {
        int3 res;
        decode(cid, res.x, res.y, res.z);
        return res;
    }

    template<CellListsProjection Projection = CellListsProjection::Clamp>
    __device__ __host__ inline int3 getCellIdAlongAxes(const float3 x) const
    {
        const int3 v = make_int3( floorf(invh * (x + 0.5f*localDomainSize)) );

        if (Projection == CellListsProjection::Clamp)
            return min( ncells - 1, max(make_int3(0), v) );
        else
            return v;
    }

    template<CellListsProjection Projection = CellListsProjection::Clamp, typename T>
    __device__ __host__ inline int getCellId(const T coo) const
    {
        const int3 id = getCellIdAlongAxes<CellListsProjection::Clamp>(make_float3(coo));

        if (Projection == CellListsProjection::NoClamp)
        {
            if (id.x < 0 || id.x >= ncells.x  ||  id.y < 0 || id.y >= ncells.y  ||  id.z < 0 || id.z >= ncells.z)
                return -1;
        }

        return encode(id.x, id.y, id.z);
    }
#endif
};

class CellList : public CellListInfo
{
protected:
    int changedStamp{-1};

    DeviceBuffer<char> scanBuffer;
    PinnedBuffer<Particle> particlesContainer = {};
    DeviceBuffer<Force>    forcesContainer = {};

    ParticleVector* pv;

    void _build(cudaStream_t stream);

    PinnedBuffer<Particle>* particles;
    DeviceBuffer<Force>*    forces;

public:

    DeviceBuffer<int> cellStarts, cellSizes, order;

    CellList(ParticleVector* pv, float rc, float3 localDomainSize);
    CellList(ParticleVector* pv, int3 resolution, float3 localDomainSize);

    CellListInfo cellInfo();

    virtual void build(cudaStream_t stream);
    virtual void addForces(cudaStream_t stream);
    void clearForces(cudaStream_t stream);

    void setViewPtrs(PVview& view);

    virtual ~CellList() = default;
};

class PrimaryCellList : public CellList
{
public:

    PrimaryCellList(ParticleVector* pv, float rc, float3 localDomainSize);
    PrimaryCellList(ParticleVector* pv, int3 resolution, float3 localDomainSize);

    void build(cudaStream_t stream);
    void addForces(cudaStream_t stream) {};

    ~PrimaryCellList() = default;
};


