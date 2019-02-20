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
    float3 localDomainSize;
    float3 h, invh;
    float rc;

    int *cellSizes, *cellStarts, *order;

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
public:    

    enum class ExtraChannelRole {IntermediateOutput, IntermediateInput, FinalOutput, None};
    using ActivePredicate = std::function<bool()>;
    
    CellList(ParticleVector *pv, float rc, float3 localDomainSize);
    CellList(ParticleVector *pv, int3 resolution, float3 localDomainSize);

    virtual ~CellList();
    
    CellListInfo cellInfo();

    virtual void build(cudaStream_t stream);
    virtual void accumulateInteractionOutput(cudaStream_t stream);
    virtual void accumulateInteractionIntermediate(cudaStream_t stream);

    virtual void gatherInteractionIntermediate(cudaStream_t stream);
    
    void clearInteractionOutput(cudaStream_t stream);
    void clearInteractionIntermediate(cudaStream_t stream);
    
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
    void requireExtraDataPerParticle(const std::string& name, ExtraChannelRole kind, ActivePredicate pred = [](){return true;})
    {
        particlesDataContainer->extraPerParticle.createData<T>(name);

        _addToChannel(name, kind, pred);
    }

    std::vector<std::string> getInteractionOutputNames() const;
    std::vector<std::string> getInteractionIntermediateNames() const;

    void setNeededForOutput();
    void setNeededForIntermediate();
    
    bool isNeededForOutput() const;
    bool isNeededForIntermediate() const;

    LocalParticleVector* getLocalParticleVector();
    
protected:
    int changedStamp{-1};

    DeviceBuffer<char> scanBuffer;
    DeviceBuffer<int> cellStarts, cellSizes, order;

    std::unique_ptr<LocalParticleVector> particlesDataContainer;
    LocalParticleVector *localPV; // will point to particlesDataContainer or pv->local() if Primary
    
    ParticleVector* pv;

    bool _checkNeedBuild() const;
    void _updateExtraDataChannels(cudaStream_t stream);
    void _computeCellSizes(cudaStream_t stream);
    void _computeCellStarts(cudaStream_t stream);
    void _reorderData(cudaStream_t stream);
    void _reorderPersistentData(cudaStream_t stream);
    
    void _build(cudaStream_t stream);
    
    /**
     *  structure to describe which channels are to be reordered, cleared and accumulated
     */
    struct ChannelActivity
    {
        std::string name;
        ActivePredicate active;
    };
    
    std::vector<ChannelActivity> finalOutputChannels;        ///< channels which are final output of interactions, e.g. forces, stresses for dpd kernel
    std::vector<ChannelActivity> intermediateOutputChannels; ///< channels which are intermediate output of interactions, e.g. densities for density kernel
    std::vector<ChannelActivity> intermediateInputChannels;  ///< channels which are intermediate input for interactions, e.g. densities for mdpd kernel

    void _addIfNameNoIn(const std::string& name, CellList::ActivePredicate pred, std::vector<ChannelActivity>& vec) const;
    void _addToChannel(const std::string& name, ExtraChannelRole kind, ActivePredicate pred);

    bool neededForOutput {false};
    bool neededForIntermediate {false};
    
    void _accumulateExtraData(std::vector<ChannelActivity>& channels, cudaStream_t stream);
    void _reorderExtraDataEntry(const std::string& channelName,
                                const ExtraDataManager::ChannelDescription *channelDesc,
                                cudaStream_t stream);

    virtual std::string makeName() const;
};

class PrimaryCellList : public CellList
{
public:

    PrimaryCellList(ParticleVector *pv, float rc, float3 localDomainSize);
    PrimaryCellList(ParticleVector *pv, int3 resolution, float3 localDomainSize);

    ~PrimaryCellList();
    
    void build(cudaStream_t stream);
    void accumulateInteractionOutput(cudaStream_t stream) override;
    void accumulateInteractionIntermediate(cudaStream_t stream) override;

    void gatherInteractionIntermediate(cudaStream_t stream) override;

protected:

    void _swapPersistentExtraData();
    std::string makeName() const override;
};


