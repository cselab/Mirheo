#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/pvs/object_vector.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/typeMap.h>
#include <core/logger.h>

#include <extern/cub/cub/device/device_scan.cuh>

namespace CellListKernels
{

enum {INVALID = -1};

inline __device__ bool outgoingParticle(float4 pos)
{
    return Float3_int(pos).isMarked();
}

__global__ void computeCellSizes(PVview view, CellListInfo cinfo)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    float4 coo = readNoCache(view.particles + pid*2);
    int cid = cinfo.getCellId(coo);

    // XXX: relying here only on redistribution
    if ( !outgoingParticle(coo) )
        atomicAdd(cinfo.cellSizes + cid, 1);
}

__global__ void reorderParticles(PVview view, CellListInfo cinfo, float4 *outParticles)
{
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int pid = gid / 2;
    const int sh  = gid % 2;  // sh = 0 copies coordinates, sh = 1 -- velocity
    if (pid >= view.size) return;

    int dstId;

    // this is to allow more cache for atomics
    // loads / stores here need no cache
    float4 val = readNoCache(view.particles+gid);

    int cid;
    if (sh == 0)
    {
        cid = cinfo.getCellId(val);

        //  XXX: relying here only on redistribution
        if ( !outgoingParticle(val) )
            dstId = cinfo.cellStarts[cid] + atomicAdd(cinfo.cellSizes + cid, 1);
        else
            dstId = -1;
    }

    int otherDst = warpShflUp(dstId, 1);
    if (sh == 1)
        dstId = otherDst;

    if (dstId >= 0)
    {
        writeNoCache(outParticles + 2*dstId+sh, val);
        if (sh == 0) cinfo.order[pid] = dstId;
    }
    else if (sh == 0)
        cinfo.order[pid] = INVALID;
}

template <typename T>
__global__ void reorderExtraDataPerParticle(int n, const T *inExtraData, CellListInfo cinfo, T *outExtraData)
{
    int srcId = blockIdx.x * blockDim.x + threadIdx.x;
    if (srcId >= n) return;

    int dstId = cinfo.order[srcId];
    if (dstId != INVALID)
        outExtraData[dstId] = inExtraData[srcId];
}

__global__ void addForcesKernel(PVview dstView, CellListInfo cinfo, PVview srcView)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= dstView.size) return;

    int srcId = cinfo.order[pid];

    assert(srcId != INVALID);
    dstView.forces[pid] += srcView.forces[srcId];
}

template <typename T>
__global__ void accumulateKernel(int n, T *dst, CellListInfo cinfo, const T *src)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n) return;

    int srcId = cinfo.order[pid];

    assert(srcId != INVALID);
    dst[pid] += src[srcId];
}

} // namespace CellListKernels

//=================================================================================
// Info
//=================================================================================

CellListInfo::CellListInfo(float rc, float3 localDomainSize) :
        rc(rc), h(make_float3(rc)), localDomainSize(localDomainSize)
{
    ncells = make_int3( floorf(localDomainSize / rc + 1e-6) );
    float3 h = make_float3(localDomainSize) / make_float3(ncells);
    invh = 1.0f / h;
    this->rc = std::min( {h.x, h.y, h.z} );

    totcells = ncells.x * ncells.y * ncells.z;
}

CellListInfo::CellListInfo(float3 h, float3 localDomainSize) :
        h(h), invh(1.0f/h), localDomainSize(localDomainSize)
{
    rc = std::min( {h.x, h.y, h.z} );
    ncells = make_int3( ceilf(localDomainSize / h - 1e-6f) );
    totcells = ncells.x * ncells.y * ncells.z;
}

//=================================================================================
// Basic cell-lists
//=================================================================================

CellList::CellList(ParticleVector *pv, float rc, float3 localDomainSize) :
        CellListInfo(rc, localDomainSize), pv(pv),
        particlesDataContainer(new LocalParticleVector(nullptr))
{
    localPV = particlesDataContainer.get();
    
    cellSizes. resize_anew(totcells + 1);
    cellStarts.resize_anew(totcells + 1);

    cellSizes. clear(0);
    cellStarts.clear(0);
    CUDA_Check( cudaStreamSynchronize(0) );

    debug("Initialized %s cell-list with %dx%dx%d cells and cut-off %f", pv->name.c_str(), ncells.x, ncells.y, ncells.z, this->rc);
}

CellList::CellList(ParticleVector *pv, int3 resolution, float3 localDomainSize) :
        CellListInfo(localDomainSize / make_float3(resolution), localDomainSize), pv(pv),
        particlesDataContainer(new LocalParticleVector(nullptr))
{
    localPV = particlesDataContainer.get();
    
    cellSizes. resize_anew(totcells + 1);
    cellStarts.resize_anew(totcells + 1);

    cellSizes. clear(0);
    cellStarts.clear(0);
    CUDA_Check( cudaStreamSynchronize(0) );

    debug("Initialized %s cell-list with %dx%dx%d cells and cut-off %f", pv->name.c_str(), ncells.x, ncells.y, ncells.z, this->rc);
}

CellList::~CellList() = default;

bool CellList::_checkNeedBuild() const
{
    if (changedStamp == pv->cellListStamp)
    {
        debug2("%s is already up-to-date, building skipped", makeName().c_str());
        return false;
    }

    if (pv->local()->size() == 0)
    {
        debug2("%s consists of no particles, building skipped", makeName().c_str());
        return false;
    }

    return true;
}

template <typename T>
static void requireData(const std::string& channelName, int np, ExtraDataManager& containerManager)
{
    if (!containerManager.checkChannelExists(channelName))
        containerManager.createData<T>(channelName, np);
}


void CellList::_updateExtraDataChannels(cudaStream_t stream)
{
    auto& pvManager        = pv->local()->extraPerParticle;
    auto& containerManager = particlesDataContainer->extraPerParticle;
    int np = pv->local()->size();

    for (const auto& namedChannel : pvManager.getSortedChannels()) {
        const auto& name = namedChannel.first;
        const auto& desc = namedChannel.second;
        if (desc->persistence != ExtraDataManager::PersistenceMode::Persistent) continue;

#define SWITCH_ENTRY(ctype)                                             \
        case DataType::TOKENIZE(ctype):                                 \
            requireData<ctype>(name, np, containerManager);             \
            break;

        switch(desc->dataType) {
            TYPE_TABLE(SWITCH_ENTRY);
        default:
            die("%s: cannot require extra data: %s has None type.", makeName().c_str(), name.c_str());
        }

#undef SWITCH_ENTRY        
    }
}

void CellList::_computeCellSizes(cudaStream_t stream)
{
    debug2("%s : Computing cell sizes for %d particles", makeName().c_str(), pv->local()->size());
    cellSizes.clear(stream);

    PVview view(pv, pv->local());

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            CellListKernels::computeCellSizes,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, cellInfo() );
}

void CellList::_computeCellStarts(cudaStream_t stream)
{
    size_t bufSize;
    cub::DeviceScan::ExclusiveSum(nullptr, bufSize, cellSizes.devPtr(), cellStarts.devPtr(), totcells+1, stream);
    scanBuffer.resize_anew(bufSize);
    cub::DeviceScan::ExclusiveSum(scanBuffer.devPtr(), bufSize, cellSizes.devPtr(), cellStarts.devPtr(), totcells+1, stream);
}

void CellList::_reorderData(cudaStream_t stream)
{
    debug2("Reordering %d %s particles", pv->local()->size(), pv->name.c_str());

    PVview view(pv, pv->local());

    order.resize_anew(view.size);
    particlesDataContainer->resize_anew(view.size);
    cellSizes.clear(stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
        CellListKernels::reorderParticles,
        getNblocks(2*view.size, nthreads), nthreads, 0, stream,
        view, cellInfo(), (float4*)particlesDataContainer->coosvels.devPtr() );
}

template <typename T>
static void reorderExtraDataEntry(int np, CellListInfo cinfo, ExtraDataManager *dstExtraData,
                                  const ExtraDataManager::ChannelDescription *channel, const std::string& channelName,
                                  cudaStream_t stream)
{
    T      *outExtraData = dstExtraData->getData<T>(channelName)->devPtr();
    const T *inExtraData = (const T*) channel->container->genericDevPtr();

    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        CellListKernels::reorderExtraDataPerParticle<T>,
        getNblocks(np, nthreads), nthreads, 0, stream,
        np, inExtraData, cinfo, outExtraData );
}

void CellList::_reorderExtraDataEntry(const std::string& channelName,
                                      const ExtraDataManager::ChannelDescription *channelDesc,
                                      cudaStream_t stream)
{
    auto dstExtraData = &particlesDataContainer->extraPerParticle;
    int np = pv->local()->size();
    
    switch (channelDesc->dataType)
    {

#define SWITCH_ENTRY(ctype)                             \
        case DataType::TOKENIZE(ctype):                 \
            reorderExtraDataEntry<ctype>                \
                (np, cellInfo(), dstExtraData,          \
                 channelDesc, channelName, stream);     \
            break;

        TYPE_TABLE(SWITCH_ENTRY);

#undef SWITCH_ENTRY

    default:
        die("%s : cannot reorder data: channel '%s' has None type",
            makeName().c_str(), channelName.c_str());
    };

}

void CellList::_reorderPersistentData(cudaStream_t stream)
{
    auto srcExtraData = &pv->local()->extraPerParticle;
    
    for (const auto& namedChannel : srcExtraData->getSortedChannels()) {
        const auto& name = namedChannel.first;
        const auto& desc = namedChannel.second;
        if (desc->persistence != ExtraDataManager::PersistenceMode::Persistent) continue;
        _reorderExtraDataEntry(name, desc, stream);
    }
}

void CellList::_build(cudaStream_t stream)
{
    _computeCellSizes(stream);
    _computeCellStarts(stream);
    _reorderData(stream);
    _reorderPersistentData(stream);
    
    changedStamp = pv->cellListStamp;
}

CellListInfo CellList::cellInfo()
{
    CellListInfo::cellSizes  = cellSizes.devPtr();
    CellListInfo::cellStarts = cellStarts.devPtr();
    CellListInfo::order      = order.devPtr();

    return *((CellListInfo*)this);
}

void CellList::build(cudaStream_t stream)
{
    _updateExtraDataChannels(stream);
        
    if (!_checkNeedBuild()) return;
    
    debug("building %s", makeName().c_str());
    
    _build(stream);
}

void CellList::_accumulateForces(cudaStream_t stream)
{
    PVview dstView(pv, pv->local());
    int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            CellListKernels::addForcesKernel,
            getNblocks(dstView.size, nthreads), nthreads, 0, stream,
            dstView, cellInfo(), getView<PVview>() );
}

void CellList::_accumulateExtraData(const std::string& channelName, cudaStream_t stream)
{
    const int nthreads = 128;
    switch(localPV->extraPerParticle.getChannelDescOrDie(channelName).dataType) {

#define SWITCH_ENTRY(ctype)                                             \
        case DataType::TOKENIZE(ctype):                                 \
        {                                                               \
            auto src = localPV    ->extraPerParticle.getData<ctype>(channelName); \
            auto dst = pv->local()->extraPerParticle.getData<ctype>(channelName); \
            int n = pv->local()->size();                                \
            SAFE_KERNEL_LAUNCH(                                         \
                CellListKernels::accumulateKernel<ctype>,               \
                getNblocks(n, nthreads), nthreads, 0, stream,           \
                n, dst->devPtr(), cellInfo(), src->devPtr() );          \
            }                                                           \
            break;

        TYPE_TABLE_ADDITIONABLE(SWITCH_ENTRY);

    default:
        die("%s : cannot accumulate entry '%s': type not supported",
            makeName().c_str(), channelName.c_str());

#undef SWITCH_ENTRY
    };        
}

void CellList::_accumulateExtraData(std::vector<ChannelActivity>& channels, cudaStream_t stream)
{   
    for (auto& entry : channels) {
        if (!entry.active()) continue;

        debug("%s : accumulating channel '%s'",
              makeName().c_str(), entry.name.c_str(), pv->name.c_str(), rc);

        _accumulateExtraData(entry.name, stream);
    }    
}

void CellList::accumulateInteractionOutput(cudaStream_t stream)
{
    _accumulateForces(stream);
    _accumulateExtraData(finalOutputChannels, stream);
}

void CellList::accumulateInteractionIntermediate(cudaStream_t stream)
{
    _accumulateExtraData(intermediateOutputChannels, stream);
}

void CellList::gatherInteractionIntermediate(cudaStream_t stream)
{
    for (auto& entry : intermediateInputChannels) {
        if (!entry.active()) continue;

        debug("%s : gathering intermediate channel '%s'",
              makeName().c_str(), entry.name.c_str());
        
        auto& desc = localPV->extraPerParticle.getChannelDescOrDie(entry.name);
        _reorderExtraDataEntry(entry.name, &desc, stream);

        // invalidate particle vector halo if any entry is active
        pv->haloValid = false;
    }
}

void CellList::clearInteractionOutput(cudaStream_t stream)
{
    localPV->forces.clear(stream);

    for (auto& channel : finalOutputChannels) {
        if (!channel.active()) continue;
        localPV->extraPerParticle.getGenericData(channel.name)->clearDevice(stream);
    }
}

void CellList::clearInteractionIntermediate(cudaStream_t stream)
{
    for (const auto& channel : intermediateInputChannels) {
        if (!channel.active()) continue;
        debug2("%s : clear channel '%s'", makeName().c_str(), channel.name.c_str());
        localPV->extraPerParticle.getGenericData(channel.name)->clearDevice(stream);
    }
    for (const auto& channel : intermediateOutputChannels) {
        if (!channel.active()) continue;
        debug2("%s : clear channel '%s'", makeName().c_str(), channel.name.c_str());
        localPV->extraPerParticle.getGenericData(channel.name)->clearDevice(stream);
    }
}

void CellList::accumulateChannels(const std::vector<std::string>& channelNames, cudaStream_t stream)
{
    for (const auto& channelName : channelNames) {
        debug2("%s : accumulating channel '%s'", makeName().c_str(), channelName.c_str());

        if (channelName == ChannelNames::forces)
            _accumulateForces(stream);
        else
            _accumulateExtraData(channelName, stream);
    }
}

void CellList::gatherChannels(const std::vector<std::string>& channelNames, cudaStream_t stream)
{
    for (auto& channelName : channelNames) {

        debug("%s : gathering channel '%s'", makeName().c_str(), channelName.c_str());
        
        auto& desc = localPV->extraPerParticle.getChannelDescOrDie(channelName);
        _reorderExtraDataEntry(channelName, &desc, stream);

        // invalidate particle vector halo if any entry is active
        pv->haloValid = false;
    }
}

void CellList::clearChannels(const std::vector<std::string>& channelNames, cudaStream_t stream)
{
    for (const auto& channelName : channelNames) {
        debug2("%s : clearing channel '%s'", makeName().c_str(), channelName.c_str());

        if (channelName == ChannelNames::forces)
            localPV->forces.clear(stream);
        else
            localPV->extraPerParticle.getGenericData(channelName)->clearDevice(stream);
    }
}

std::vector<std::string> CellList::getInteractionOutputNames() const
{
    std::vector<std::string> names;
    for (const auto& entry : finalOutputChannels)
        names.push_back(entry.name);
    return names;
}

std::vector<std::string> CellList::getInteractionIntermediateNames() const
{
    std::vector<std::string> names;
    for (const auto& entry : intermediateOutputChannels)
        names.push_back(entry.name);
    return names;
}

void CellList::setNeededForOutput() {neededForOutput = true;}
void CellList::setNeededForIntermediate() {neededForIntermediate = true;}

bool CellList::isNeededForOutput() const {return neededForOutput;}
bool CellList::isNeededForIntermediate() const {return neededForIntermediate;}

LocalParticleVector* CellList::getLocalParticleVector() {return localPV;}

void CellList::_addIfNameNoIn(const std::string& name, CellList::ActivePredicate pred, std::vector<CellList::ChannelActivity>& vec) const
{
    bool alreadyIn = false;
    for (const auto& entry : vec)
        if (entry.name == name)
            alreadyIn = true;

    if (alreadyIn) {
        debug("%s : channel '%s' already added, skip it. Make sure that the activity predicate is the same",
              makeName().c_str(), name.c_str());
        // We could also make pred = old_pred || pred; leave it as it is for now
        return;
    }
    
    vec.push_back({name, pred});
}

void CellList::_addToChannel(const std::string& name, ExtraChannelRole kind, CellList::ActivePredicate pred)
{
    debug("%s : adding channel %s", makeName().c_str(), name.c_str());
    if      (kind == ExtraChannelRole::IntermediateOutput) _addIfNameNoIn(name, pred, intermediateOutputChannels);
    else if (kind == ExtraChannelRole::IntermediateInput)  _addIfNameNoIn(name, pred, intermediateInputChannels);
    else if (kind == ExtraChannelRole::FinalOutput)        _addIfNameNoIn(name, pred, finalOutputChannels);
}

std::string CellList::makeName() const
{
    return "Cell List '" + pv->name + "' (rc " + std::to_string(rc) + ")";
}


//=================================================================================
// Primary cell-lists
//=================================================================================

PrimaryCellList::PrimaryCellList(ParticleVector *pv, float rc, float3 localDomainSize) :
        CellList(pv, rc, localDomainSize)
{
    localPV = pv->local();

    if (dynamic_cast<ObjectVector*>(pv) != nullptr)
        error("Using primary cell-lists with objects is STRONGLY discouraged. This will very likely result in an error");
}

PrimaryCellList::PrimaryCellList(ParticleVector *pv, int3 resolution, float3 localDomainSize) :
        CellList(pv, resolution, localDomainSize)
{
    localPV = pv->local();

    if (dynamic_cast<ObjectVector*>(pv) != nullptr)
        error("Using primary cell-lists with objects is STRONGLY discouraged. This will very likely result in an error");
}

PrimaryCellList::~PrimaryCellList() = default;

void PrimaryCellList::build(cudaStream_t stream)
{
    CellList::build(stream);

    if (pv->local()->size() == 0)
    {
        debug2("%s consists of no particles, cell-list building skipped", pv->name.c_str());
        return;
    }
    
    // Now we need the new size of particles array.
    int newSize;
    CUDA_Check( cudaMemcpyAsync(&newSize, cellStarts.devPtr() + totcells, sizeof(int), cudaMemcpyDeviceToHost, stream) );
    CUDA_Check( cudaStreamSynchronize(stream) );

    debug2("%s : reordering completed, new size of %s particle vector is %d",
           makeName().c_str(), pv->name.c_str(), newSize);

    particlesDataContainer->resize(newSize, stream);

    std::swap(pv->local()->coosvels, particlesDataContainer->coosvels);
    _swapPersistentExtraData();
    
    pv->local()->resize(newSize, stream);
}

void PrimaryCellList::accumulateInteractionOutput(cudaStream_t stream)
{}

void PrimaryCellList::accumulateInteractionIntermediate(cudaStream_t stream)
{}    

void PrimaryCellList::gatherInteractionIntermediate(cudaStream_t stream)
{    
    // do not need to reorder data, but still invalidate halo
    for (auto& entry : intermediateInputChannels) {
        if (!entry.active()) continue;
        pv->haloValid = false;
    }
}

void PrimaryCellList::accumulateChannels(const std::vector<std::string>& channelNames, cudaStream_t stream)
{}

void PrimaryCellList::gatherChannels(const std::vector<std::string>& channelNames, cudaStream_t stream)
{
    // do not need to reorder data, but still invalidate halo
    if (!channelNames.empty())
        pv->haloValid = false;
}


template <typename T>
static void swap(const std::string& channelName, ExtraDataManager& pvManager, ExtraDataManager& containerManager)
{
    std::swap(*pvManager       .getData<T>(channelName),
              *containerManager.getData<T>(channelName));
}

void PrimaryCellList::_swapPersistentExtraData()
{
    auto& pvManager        = pv->local()->extraPerParticle;
    auto& containerManager = particlesDataContainer->extraPerParticle;
    
    for (const auto& namedChannel : pvManager.getSortedChannels()) {
        const auto& name = namedChannel.first;
        const auto& desc = namedChannel.second;
        if (desc->persistence != ExtraDataManager::PersistenceMode::Persistent) continue;

#define SWITCH_ENTRY(ctype)                                             \
        case DataType::TOKENIZE(ctype):                                 \
            swap<ctype>(name, pvManager, containerManager);             \
            break;

        switch(desc->dataType) {
            TYPE_TABLE(SWITCH_ENTRY);
        default:
            die("%s: cannot swap data: %s has None type.",
                makeName().c_str(), name.c_str());
        }

#undef SWITCH_ENTRY        
    }
}

std::string PrimaryCellList::makeName() const
{
    return "Primary " + CellList::makeName();
}
