#include "celllist.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/type_traits.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <extern/cub/cub/device/device_scan.cuh>
#pragma GCC diagnostic pop

#include <algorithm>

namespace mirheo
{

namespace cell_list_kernels
{

enum {INVALID = -1};

inline __device__ bool outgoingParticle(real4 pos)
{
    return Real3_int(pos).isMarked();
}

__global__ void computeCellSizes(PVview view, CellListInfo cinfo)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    real4 coo = view.readPositionNoCache(pid);

    // XXX: relying here only on redistribution
    if ( outgoingParticle(coo) ) return;

    int cid = cinfo.getCellId<CellListsProjection::Clamp>(coo);
    atomicAdd(cinfo.cellSizes + cid, 1);
}

__global__ void reorderPositionsAndCreateMap(PVview view, CellListInfo cinfo, real4 *outPositions)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    int dstId = INVALID;

    // this is to allow more cache for atomics
    // loads / stores here need no cache
    real4 pos = view.readPositionNoCache(pid);

    int cid = cinfo.getCellId<CellListsProjection::Clamp>(pos);

    //  XXX: relying here only on redistribution
    if ( !outgoingParticle(pos) )
        dstId = cinfo.cellStarts[cid] + atomicAdd(cinfo.cellSizes + cid, 1);

    if (dstId != INVALID)
        writeNoCache(outPositions + dstId, pos);

    cinfo.order[pid] = dstId;
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

template <typename T>
__global__ void accumulateKernel(int n, T *dst, CellListInfo cinfo, const T *src)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n) return;

    int srcId = cinfo.order[pid];

    assert(srcId != INVALID);
    dst[pid] += src[srcId];
}

} // namespace cell_list_kernels

//=================================================================================
// Info
//=================================================================================

CellListInfo::CellListInfo(real rc_, real3 localDomainSize_) :
    rc(rc_),
    localDomainSize(localDomainSize_)
{
    ncells = make_int3( math::floor(localDomainSize / rc + 1e-6_r) );
    h = make_real3(localDomainSize) / make_real3(ncells);
    invh_ = 1.0_r / h;
    rc = std::min( {h.x, h.y, h.z} );

    totcells = ncells.x * ncells.y * ncells.z;
}

CellListInfo::CellListInfo(real3 h_, real3 localDomainSize_) :
    h(h_),
    invh_(1.0_r/h_),
    localDomainSize(localDomainSize_)
{
    ncells = make_int3( math::ceil(localDomainSize / h - 1e-6_r) );
    totcells = ncells.x * ncells.y * ncells.z;
    h = make_real3(localDomainSize) / make_real3(ncells); // in case h does not divide localDomainSize
    invh_ = 1.0_r / h;
    rc = std::min( {h.x, h.y, h.z} );
}

//=================================================================================
// Basic cell-lists
//=================================================================================

CellList::CellList(ParticleVector *pv, real rc_, real3 localDomainSize_) :
    CellListInfo(rc_, localDomainSize_),
    pv_(pv),
    particlesDataContainer_(std::make_unique<LocalParticleVector>(nullptr))
{
    _initialize();
}

CellList::CellList(ParticleVector *pv, int3 resolution, real3 localDomainSize_) :
    CellListInfo(localDomainSize_ / make_real3(resolution), localDomainSize_),
    pv_(pv),
    particlesDataContainer_(std::make_unique<LocalParticleVector>(nullptr))
{
    _initialize();
}

void CellList::_initialize()
{
    localPV_ = particlesDataContainer_.get();

    cellSizes. resize_anew(totcells + 1);
    cellStarts.resize_anew(totcells + 1);

    cellSizes. clear(defaultStream);
    cellStarts.clear(defaultStream);
    CUDA_Check( cudaStreamSynchronize(defaultStream) );

    debug("Initialized %s cell-list with %dx%dx%d cells and cut-off %f", pv_->getCName(), ncells.x, ncells.y, ncells.z, rc);
}

CellList::~CellList() = default;

bool CellList::_checkNeedBuild() const
{
    if (changedStamp_ == pv_->cellListStamp)
    {
        debug2("%s is already up-to-date, building skipped", _makeName().c_str());
        return false;
    }

    if (pv_->local()->size() == 0)
    {
        debug2("%s consists of no particles, building skipped", _makeName().c_str());
        return false;
    }

    return true;
}

void CellList::_updateExtraDataChannels(__UNUSED cudaStream_t stream)
{
    auto& pvManager        = pv_->local()->dataPerParticle;
    auto& containerManager = particlesDataContainer_->dataPerParticle;
    const int np = pv_->local()->size();

    for (const auto& namedChannel : pvManager.getSortedChannels())
    {
        const auto& name = namedChannel.first;
        const auto& desc = namedChannel.second;
        if (desc->persistence != DataManager::PersistenceMode::Active) continue;

        mpark::visit([&](auto pinnedBuffPtr)
        {
            using T = typename std::remove_pointer<decltype(pinnedBuffPtr)>::type::value_type;

            if (!containerManager.checkChannelExists(name))
                containerManager.createData<T>(name, np);

        }, desc->varDataPtr);
    }
}

void CellList::_computeCellSizes(cudaStream_t stream)
{
    debug2("%s : Computing cell sizes for %d particles", _makeName().c_str(), pv_->local()->size());
    cellSizes.clear(stream);

    PVview view(pv_, pv_->local());

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            cell_list_kernels::computeCellSizes,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, cellInfo() );
}

void CellList::_computeCellStarts(cudaStream_t stream)
{
    // Scan is always working with the same number of cells
    // Memory requirements can't change
    size_t bufSize = scanBuffer.size();

    if (bufSize == 0)
    {
        cub::DeviceScan::ExclusiveSum(nullptr, bufSize, cellSizes.devPtr(), cellStarts.devPtr(), totcells+1, stream);
        scanBuffer.resize_anew(bufSize);
    }
    cub::DeviceScan::ExclusiveSum(scanBuffer.devPtr(), bufSize,
                                  cellSizes.devPtr(), cellStarts.devPtr(), totcells+1, stream);
}

void CellList::_reorderPositionsAndCreateMap(cudaStream_t stream)
{
    debug2("Reordering %d %s particles", pv_->local()->size(), pv_->getCName());

    PVview view(pv_, pv_->local());

    order.resize_anew(view.size);
    particlesDataContainer_->resize_anew(view.size);
    cellSizes.clear(stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
        cell_list_kernels::reorderPositionsAndCreateMap,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, cellInfo(), particlesDataContainer_->positions().devPtr() );
}

void CellList::_reorderExtraDataEntry(const std::string& channelName,
                                      const DataManager::ChannelDescription *channelDesc,
                                      cudaStream_t stream)
{
    const auto& dstDesc = particlesDataContainer_->dataPerParticle.getChannelDescOrDie(channelName);
    const int np = pv_->local()->size();

    debug2("%s: reordering extra data '%s'", _makeName().c_str(), channelName.c_str());

    mpark::visit([&](auto srcPinnedBuff)
    {
        auto dstPinnedBuff = mpark::get<decltype(srcPinnedBuff)>(dstDesc.varDataPtr);

        constexpr int nthreads = 128;

        SAFE_KERNEL_LAUNCH(
           cell_list_kernels::reorderExtraDataPerParticle,
           getNblocks(np, nthreads), nthreads, 0, stream,
           np, srcPinnedBuff->devPtr(), this->cellInfo(), dstPinnedBuff->devPtr() );
    }, channelDesc->varDataPtr);
}

void CellList::_reorderPersistentData(cudaStream_t stream)
{
    auto srcExtraData = &pv_->local()->dataPerParticle;

    for (const auto& namedChannel : srcExtraData->getSortedChannels())
    {
        const auto& name = namedChannel.first;
        const auto& desc = namedChannel.second;
        if (desc->persistence != DataManager::PersistenceMode::Active
            || name == channel_names::positions) // positions were already reordered manually
            continue;
        _reorderExtraDataEntry(name, desc, stream);
    }
}

void CellList::_build(cudaStream_t stream)
{
    _computeCellSizes(stream);
    _computeCellStarts(stream);
    _reorderPositionsAndCreateMap(stream);
    _reorderPersistentData(stream);

    changedStamp_ = pv_->cellListStamp;
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

    debug("building %s", _makeName().c_str());

    _build(stream);
}

static void accumulateIfHasAddOperator(__UNUSED GPUcontainer *src,
                                       __UNUSED GPUcontainer *dst,
                                       __UNUSED int n, __UNUSED CellListInfo cinfo,
                                       __UNUSED cudaStream_t stream)
{
    die("Cannot accumulate entries: operator+ not supported for this type");
}

// use SFINAE to choose between additionable types
template <typename T, typename = void_t<decltype(std::declval<T>() +
                                                 std::declval<T>())>>
static void accumulateIfHasAddOperator(PinnedBuffer<T> *src,
                                       PinnedBuffer<T> *dst,
                                       int n, CellListInfo cinfo,
                                       cudaStream_t stream)
{
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        cell_list_kernels::accumulateKernel,
        getNblocks(n, nthreads), nthreads, 0, stream,
        n, dst->devPtr(), cinfo, src->devPtr() );
}

void CellList::_accumulateExtraData(const std::string& channelName, cudaStream_t stream)
{
    const int n = pv_->local()->size();

    const auto& pvManager   = pv_->local()->dataPerParticle;
    const auto& contManager = localPV_->dataPerParticle;

    const auto& pvDesc   = pvManager  .getChannelDescOrDie(channelName);
    const auto& contDesc = contManager.getChannelDescOrDie(channelName);

    mpark::visit([&](auto srcPinnedBuff)
    {
        auto dstPinnedBuff = mpark::get<decltype(srcPinnedBuff)>(pvDesc.varDataPtr);
        accumulateIfHasAddOperator(srcPinnedBuff, dstPinnedBuff, n, this->cellInfo(), stream);
    }, contDesc.varDataPtr);
}

void CellList::accumulateChannels(const std::vector<std::string>& channelNames, cudaStream_t stream)
{
    for (const auto& channelName : channelNames)
    {
        debug2("%s : accumulating channel '%s'", _makeName().c_str(), channelName.c_str());
        _accumulateExtraData(channelName, stream);
    }
}

void CellList::gatherChannels(const std::vector<std::string>& channelNames, cudaStream_t stream)
{
    for (auto& channelName : channelNames)
    {
        debug("%s : gathering channel '%s'", _makeName().c_str(), channelName.c_str());

        auto& desc = localPV_->dataPerParticle.getChannelDescOrDie(channelName);
        _reorderExtraDataEntry(channelName, &desc, stream);

        // invalidate particle vector halo if any entry is active
        pv_->haloValid = false;
    }
}

void CellList::clearChannels(const std::vector<std::string>& channelNames, cudaStream_t stream)
{
    for (const auto& channelName : channelNames)
    {
        debug2("%s : clearing channel '%s'", _makeName().c_str(), channelName.c_str());
        localPV_->dataPerParticle.getGenericData(channelName)->clearDevice(stream);
    }
}

LocalParticleVector* CellList::getLocalParticleVector() {return localPV_;}

std::string CellList::_makeName() const
{
    return "Cell List '" + pv_->getName() + "' (rc " + std::to_string(rc) + ")";
}


//=================================================================================
// Primary cell-lists
//=================================================================================

PrimaryCellList::PrimaryCellList(ParticleVector *pv, real rc_, real3 localDomainSize_) :
        CellList(pv, rc_, localDomainSize_)
{
    localPV_ = pv_->local();

    if (dynamic_cast<ObjectVector*>(pv_) != nullptr)
        error("Using primary cell-lists with objects is STRONGLY discouraged. This will very likely result in an error");
}

PrimaryCellList::PrimaryCellList(ParticleVector *pv, int3 resolution, real3 localDomainSize_) :
        CellList(pv, resolution, localDomainSize_)
{
    localPV_ = pv_->local();

    if (dynamic_cast<ObjectVector*>(pv_) != nullptr)
        error("Using primary cell-lists with objects is STRONGLY discouraged. This will very likely result in an error");
}

PrimaryCellList::~PrimaryCellList() = default;

void PrimaryCellList::build(cudaStream_t stream)
{
    // Reqired here to avoid ptr swap if building didn't actually happen
    if (!_checkNeedBuild()) return;

    CellList::build(stream);

    if (pv_->local()->size() == 0)
    {
        debug2("%s consists of no particles, cell-list building skipped", pv_->getCName());
        return;
    }

    // Now we need the new size of particles array.
    int newSize;
    CUDA_Check( cudaMemcpyAsync(&newSize, cellStarts.devPtr() + totcells, sizeof(int), cudaMemcpyDeviceToHost, stream) );
    CUDA_Check( cudaStreamSynchronize(stream) );

    debug2("%s : reordering completed, new size of %s particle vector is %d",
           _makeName().c_str(), pv_->getCName(), newSize);

    particlesDataContainer_->resize(newSize, stream);

    _swapPersistentExtraData();

    pv_->local()->resize(newSize, stream);
}

void PrimaryCellList::accumulateChannels(__UNUSED const std::vector<std::string>& channelNames, __UNUSED cudaStream_t stream)
{}

void PrimaryCellList::gatherChannels(const std::vector<std::string>& channelNames, __UNUSED cudaStream_t stream)
{
    // do not need to reorder data, but still invalidate halo
    if (!channelNames.empty())
        pv_->haloValid = false;
}


template <typename T>
static void swap(const std::string& channelName, DataManager& pvManager, DataManager& containerManager)
{
    std::swap(*pvManager       .getData<T>(channelName),
              *containerManager.getData<T>(channelName));
}

void PrimaryCellList::_swapPersistentExtraData()
{
    auto& pvManager        = pv_->local()->dataPerParticle;
    auto& containerManager = particlesDataContainer_->dataPerParticle;

    for (const auto& namedChannel : pvManager.getSortedChannels())
    {
        const auto& name = namedChannel.first;
        const auto& desc = namedChannel.second;
        if (desc->persistence != DataManager::PersistenceMode::Active)
            continue;

        const auto& descCont = containerManager.getChannelDescOrDie(name);

        mpark::visit([&](auto pinnedBufferPv)
        {
            auto pinnedBufferCont = mpark::get<decltype(pinnedBufferPv)>(descCont.varDataPtr);
            std::swap(*pinnedBufferPv, *pinnedBufferCont);
        }, desc->varDataPtr);
    }
}

std::string PrimaryCellList::_makeName() const
{
    return "Primary " + CellList::_makeName();
}

} // namespace mirheo
