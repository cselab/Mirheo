#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/pvs/object_vector.h>
#include <core/celllist.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
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

// TODO: use old_particles as buffer
__global__ void reorderParticles(PVview view, CellListInfo cinfo, float4* outParticles)
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

__global__ void addForcesKernel(PVview view, CellListInfo cinfo)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    view.forces[pid] += cinfo.forces[cinfo.order[pid]];
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

CellList::CellList(ParticleVector* pv, float rc, float3 localDomainSize) :
        CellListInfo(rc, localDomainSize), pv(pv),
        particles(&particlesContainer),
        forces(&forcesContainer)
{
    cellSizes. resize_anew(totcells + 1);
    cellStarts.resize_anew(totcells + 1);

    cellSizes. clear(0);
    cellStarts.clear(0);
    CUDA_Check( cudaStreamSynchronize(0) );

    debug("Initialized %s cell-list with %dx%dx%d cells and cut-off %f", pv->name.c_str(), ncells.x, ncells.y, ncells.z, this->rc);
}

CellList::CellList(ParticleVector* pv, int3 resolution, float3 localDomainSize) :
        CellListInfo(localDomainSize / make_float3(resolution), localDomainSize), pv(pv),
        particles(&particlesContainer),
        forces(&forcesContainer)
{
    cellSizes. resize_anew(totcells + 1);
    cellStarts.resize_anew(totcells + 1);

    cellSizes. clear(0);
    cellStarts.clear(0);
    CUDA_Check( cudaStreamSynchronize(0) );

    debug("Initialized %s cell-list with %dx%dx%d cells and cut-off %f", pv->name.c_str(), ncells.x, ncells.y, ncells.z, this->rc);
}


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
    particlesContainer.resize_anew(view.size);
    cellSizes.clear(stream);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
                       reorderParticles,
                       getNblocks(2*view.size, nthreads), nthreads, 0, stream,
                       view, cellInfo(), (float4*)particlesContainer.devPtr() );
}
    
void CellList::_build(cudaStream_t stream)
{
    _computeCellSizes(stream);
    _computeCellStarts(stream);
    _reorderData(stream);
    
    changedStamp = pv->cellListStamp;
}

CellListInfo CellList::cellInfo()
{
    CellListInfo::particles  = reinterpret_cast<float4*>(particles->devPtr());
    CellListInfo::forces     = reinterpret_cast<float4*>(forces->devPtr());
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

    forcesContainer.resize_anew(pv->local()->size());
}

void CellList::addForces(cudaStream_t stream)
{
    PVview view(pv, pv->local());
    int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            addForcesKernel,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, cellInfo() );
}

void CellList::clearForces(cudaStream_t stream)
{
    forces->clear(stream);
}

void CellList::setViewPtrs(PVview& view)
{
    view.particles = (float4*) particles->devPtr();
    view.forces    = (float4*) forces->devPtr();
}

//=================================================================================
// Primary cell-lists
//=================================================================================

PrimaryCellList::PrimaryCellList(ParticleVector* pv, float rc, float3 localDomainSize) :
        CellList(pv, rc, localDomainSize)
{
    particles = &pv->local()->coosvels;
    forces    = &pv->local()->forces;

    if (dynamic_cast<ObjectVector*>(pv) != nullptr)
        error("Using primary cell-lists with objects is STRONGLY discouraged. This will very likely result in an error");
}

PrimaryCellList::PrimaryCellList(ParticleVector* pv, int3 resolution, float3 localDomainSize) :
        CellList(pv, resolution, localDomainSize)
{
    particles = &pv->local()->coosvels;
    forces    = &pv->local()->forces;

    if (dynamic_cast<ObjectVector*>(pv) != nullptr)
        error("Using primary cell-lists with objects is STRONGLY discouraged. This will very likely result in an error");
}

void PrimaryCellList::build(cudaStream_t stream)
{
    //warn("Reordering extra data is not yet implemented in cell-lists");

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

    // Now we need the new size of particles array.
    int newSize;
    CUDA_Check( cudaMemcpyAsync(&newSize, cellStarts.devPtr() + totcells, sizeof(int), cudaMemcpyDeviceToHost, stream) );
    CUDA_Check( cudaStreamSynchronize(stream) );

    debug2("Reordering completed, new size of %s particle vector is %d", pv->name.c_str(), newSize);

    particlesContainer.resize(newSize, stream);
    std::swap(pv->local()->coosvels, particlesContainer);
    pv->local()->resize(newSize, stream);
}

