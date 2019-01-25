#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/pvs/object_vector.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/typeMap.h>
#include <core/logger.h>

#include <extern/cub/cub/device/device_scan.cuh>

static __device__ bool outgoingParticle(float4 pos)
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
}

template <typename T>
__global__ void reorderExtraDataPerParticle(int n, const T *inExtraData, CellListInfo cinfo, T *outExtraData)
{
    const int srcId = blockIdx.x * blockDim.x + threadIdx.x;
    if (srcId >= n) return;

    const int dstId = cinfo.order[srcId];
    outExtraData[dstId] = inExtraData[srcId];
}

__global__ void addForcesKernel(PVview dstView, CellListInfo cinfo, PVview srcView)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= dstView.size) return;

    dstView.forces[pid] += srcView.forces[cinfo.order[pid]];
}

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

void CellList::_computeCellSizes(cudaStream_t stream)
{
    debug2("Computing cell sizes for %d %s particles", pv->local()->size(), pv->name.c_str());
    cellSizes.clear(stream);

    PVview view(pv, pv->local());

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            computeCellSizes,
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
        reorderParticles,
        getNblocks(2*view.size, nthreads), nthreads, 0, stream,
        view, cellInfo(), (float4*)particlesDataContainer->coosvels.devPtr() );
}

template <typename T>
static void reorderExtraData(int np, CellListInfo cinfo, ExtraDataManager *dstExtraData,
                             const ExtraDataManager::ChannelDescription *channel, const std::string& channelName,
                             cudaStream_t stream)
{
    if (!dstExtraData->checkChannelExists(channelName))
        dstExtraData->createData<T>(channelName, np);

    T      *outExtraData = dstExtraData->getData<T>(channelName)->devPtr();
    const T *inExtraData = (const T*) channel->container->genericDevPtr();

    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
        reorderExtraDataPerParticle<T>,
        getNblocks(np, nthreads), nthreads, 0, stream,
        np, inExtraData, cinfo, outExtraData );
}

void CellList::_reorderExtraData(cudaStream_t stream)
{
    auto srcExtraData = &pv->local()->extraPerParticle;
    auto dstExtraData = &particlesDataContainer->extraPerParticle;

    int np = pv->local()->size();
    
    for (auto& namedChannel : srcExtraData->getSortedChannels())
    {
        auto channelName = namedChannel.first;
        auto channelDesc = namedChannel.second;

        if (channelDesc->persistence == ExtraDataManager::PersistenceMode::Persistent) {
            debug2("Reordering %d `%s` particles extra data `%s`",
                   pv->local()->size(), pv->name.c_str(), channelName.c_str());

            switch (channelDesc->dataType)
            {

#define SWITCH_ENTRY(ctype)                                             \
                case DataType::TOKENIZE(ctype):                         \
                    reorderExtraData<ctype>                             \
                        (np, cellInfo(), dstExtraData,                  \
                         channelDesc, channelName, stream);             \
                    break;

                TYPE_TABLE(SWITCH_ENTRY);

#undef SWITCH_ENTRY

            default:
                die("Channel '%s' has None type", channelName.c_str());
            };
        }
    }
}

void CellList::_build(cudaStream_t stream)
{
    _computeCellSizes(stream);
    _computeCellStarts(stream);
    _reorderData(stream);
    _reorderExtraData(stream);
    
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
    if (changedStamp == pv->cellListStamp)
    {
        debug2("Cell-list for %s is already up-to-date, building skipped", pv->name.c_str());
        return;
    }

    if (pv->local()->size() == 0)
    {
        debug2("%s consists of no particles, cell-list building skipped", pv->name.c_str());
        return;
    }

    _build(stream);
}

void CellList::addForces(cudaStream_t stream)
{
    PVview dstView(pv, pv->local());
    int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            addForcesKernel,
            getNblocks(dstView.size, nthreads), nthreads, 0, stream,
            dstView, cellInfo(), getView<PVview>() );
}


void CellList::clearForces(cudaStream_t stream)
{
    localPV->forces.clear(stream);
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

    // Now we need the new size of particles array.
    int newSize;
    CUDA_Check( cudaMemcpyAsync(&newSize, cellStarts.devPtr() + totcells, sizeof(int), cudaMemcpyDeviceToHost, stream) );
    CUDA_Check( cudaStreamSynchronize(stream) );

    debug2("Reordering completed, new size of %s particle vector is %d", pv->name.c_str(), newSize);

    particlesDataContainer->resize(newSize, stream);
    std::swap(pv->local()->coosvels, particlesDataContainer->coosvels);
    pv->local()->resize(newSize, stream);
}

void PrimaryCellList::addForces(cudaStream_t stream)
{}    
