#pragma once

#include <core/containers.h>
#include <core/datatypes.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cuda_common.h>

#include <cstdint>
#include <functional>


enum class CellListsProjection
{
    Clamp, NoClamp
};


class CellListInfo
{
public:
    int3 ncells;
    int  totcells;
    real3 localDomainSize;
    real3 h, invh;
    real rc;

    int *cellSizes, *cellStarts, *order;

    CellListInfo(real3 h, real3 localDomainSize);
    CellListInfo(real rc, real3 localDomainSize);

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
    __device__ __host__ inline int3 getCellIdAlongAxes(const real3 x) const
    {
        const int3 v = make_int3( math::floor(invh * (x + 0.5_r * localDomainSize)) );

        if (Projection == CellListsProjection::Clamp)
            return math::min( ncells - 1, math::max(make_int3(0), v) );
        else
            return v;
    }

    template<CellListsProjection Projection = CellListsProjection::Clamp, typename T>
    __device__ __host__ inline int getCellId(const T coo) const
    {
        const int3 id = getCellIdAlongAxes<CellListsProjection::Clamp>(make_real3(coo));

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
public:    
    
    CellList(ParticleVector *pv, real rc, real3 localDomainSize);
    CellList(ParticleVector *pv, int3 resolution, real3 localDomainSize);

    virtual ~CellList();
    
    CellListInfo cellInfo();

    virtual void build(cudaStream_t stream);

    virtual void accumulateChannels(const std::vector<std::string>& channelNames, cudaStream_t stream);
    virtual void gatherChannels(const std::vector<std::string>& channelNames, cudaStream_t stream);
    void clearChannels(const std::vector<std::string>& channelNames, cudaStream_t stream);

    
    
    template <typename ViewType>
    ViewType getView() const
    {
        return ViewType(pv, localPV);
    }

    /**
     * add extra channel to the cell-list.
     * depending on \c kind, the channel will be cleared, accumulated and scattered at different times
     * 
     */
    template <typename T>
    void requireExtraDataPerParticle(const std::string& name)
    {
        particlesDataContainer->dataPerParticle.createData<T>(name);
    }
    
    LocalParticleVector* getLocalParticleVector();
    
protected:
    int changedStamp{-1};

    DeviceBuffer<char> scanBuffer;
    DeviceBuffer<int> cellStarts, cellSizes, order;

    std::unique_ptr<LocalParticleVector> particlesDataContainer;
    LocalParticleVector *localPV; // will point to particlesDataContainer or pv->local() if Primary
    
    ParticleVector* pv;

    void _initialize();
    bool _checkNeedBuild() const;
    void _updateExtraDataChannels(cudaStream_t stream);
    void _computeCellSizes(cudaStream_t stream);
    void _computeCellStarts(cudaStream_t stream);
    void _reorderPositionsAndCreateMap(cudaStream_t stream);
    void _reorderPersistentData(cudaStream_t stream);
    
    void _build(cudaStream_t stream);
        
    void _accumulateForces(cudaStream_t stream);
    void _accumulateExtraData(const std::string& channelName, cudaStream_t stream);
    
    void _reorderExtraDataEntry(const std::string& channelName,
                                const DataManager::ChannelDescription *channelDesc,
                                cudaStream_t stream);

    virtual std::string makeName() const;
};

class PrimaryCellList : public CellList
{
public:

    PrimaryCellList(ParticleVector *pv, real rc, real3 localDomainSize);
    PrimaryCellList(ParticleVector *pv, int3 resolution, real3 localDomainSize);

    ~PrimaryCellList();
    
    void build(cudaStream_t stream);

    void accumulateChannels(const std::vector<std::string>& channelNames, cudaStream_t stream) override;
    void gatherChannels(const std::vector<std::string>& channelNames, cudaStream_t stream) override;

protected:

    void _swapPersistentExtraData();
    std::string makeName() const override;
};


