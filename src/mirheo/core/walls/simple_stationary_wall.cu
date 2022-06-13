// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "simple_stationary_wall.h"

#include "common_kernels.h"
#include "stationary_walls/box.h"
#include "stationary_walls/cylinder.h"
#include "stationary_walls/plane.h"
#include "stationary_walls/sdf.h"
#include "stationary_walls/sphere.h"
#include "velocity_field/none.h"

#include <mirheo/core/celllist.h>
#include <mirheo/core/field/utils.h>
#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/packers/objects.h>
#include <mirheo/core/pvs/object_vector.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/root_finder.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <texture_types.h>

namespace mirheo
{

enum class QueryMode {
   Query,
   Collect
};

namespace stationary_walls_kernels
{

//===============================================================================================
// Removing kernels
//===============================================================================================

template<typename InsideWallChecker>
__global__ void packRemainingParticles(PVview view, ParticlePackerHandler packer, char *outputBuffer,
                                       int *nRemaining, InsideWallChecker checker, int maxNumParticles)
{
    const real tolerance = 1e-6_r;

    const int srcPid = blockIdx.x * blockDim.x + threadIdx.x;
    if (srcPid >= view.size) return;

    const real3 r = make_real3(view.readPosition(srcPid));
    const real val = checker(r);

    if (val <= -tolerance)
    {
        const int dstPid = atomicAggInc(nRemaining);
        packer.particles.pack(srcPid, dstPid, outputBuffer, maxNumParticles);
    }
}

__global__ void unpackRemainingParticles(const char *inputBuffer, ParticlePackerHandler packer,
                                         int nRemaining, int maxNumParticles)
{
    const int srcPid = blockIdx.x * blockDim.x + threadIdx.x;
    if (srcPid >= nRemaining) return;

    const int dstPid = srcPid;
    packer.particles.unpack(srcPid, dstPid, inputBuffer, maxNumParticles);
}

template<typename InsideWallChecker>
__global__ void packRemainingObjects(OVview view, ObjectPackerHandler packer,
                                     char *output, int *nRemaining,
                                     InsideWallChecker checker, int maxNumObj)
{
    const real tolerance = 1e-6_r;

    // One warp per object
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int objId  = gid / warpSize;
    const int laneId = gid % warpSize;

    if (objId >= view.nObjects) return;

    bool isRemaining = true;
    for (int i = laneId; i < view.objSize; i += warpSize)
    {
        Particle p(view.readParticle(objId * view.objSize + i));

        if (checker(p.r) > -tolerance)
        {
            isRemaining = false;
            break;
        }
    }

    isRemaining = warpAll(isRemaining);
    if (!isRemaining) return;

    int dstObjId;
    if (laneId == 0)
        dstObjId = atomicAdd(nRemaining, 1);
    dstObjId = warpShfl(dstObjId, 0);

    size_t offsetObjData = 0;

    for (int pid = laneId; pid < view.objSize; pid += warpSize)
    {
        const int srcPid = objId    * view.objSize + pid;
        const int dstPid = dstObjId * view.objSize + pid;
        offsetObjData = packer.particles.pack(srcPid, dstPid, output, maxNumObj * view.objSize);
    }

    if (laneId == 0) packer.objects.pack(objId, dstObjId, output + offsetObjData, maxNumObj);
}

__global__ void unpackRemainingObjects(const char *from, OVview view, ObjectPackerHandler packer, int maxNumObj)
{
    const int objId = blockIdx.x;
    const int tid = threadIdx.x;

    size_t offsetObjData = 0;

    for (int pid = tid; pid < view.objSize; pid += blockDim.x)
    {
        const int dstId = objId*view.objSize + pid;
        const int srcId = objId*view.objSize + pid;
        offsetObjData = packer.particles.unpack(srcId, dstId, from, maxNumObj * view.objSize);
    }

    if (tid == 0)
        packer.objects.unpack(objId, objId, from + offsetObjData, maxNumObj);
}


//===============================================================================================
// Boundary cells kernels
//===============================================================================================

template<typename InsideWallChecker>
__device__ inline bool isCellOnBoundary(const real maximumTravel, real3 cornerCoo, real3 len, InsideWallChecker checker)
{
    int pos = 0, neg = 0;

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            for (int k = 0; k < 2; ++k)
            {
                // Value in the cell corner
                const real3 shift = make_real3(i ? len.x : 0.0_r, j ? len.y : 0.0_r, k ? len.z : 0.0_r);
                const real s = checker(cornerCoo + shift);

                if (s >  maximumTravel) pos++;
                if (s < -maximumTravel) neg++;
            }
        }
    }

    return (pos != 8 && neg != 8);
}

template<QueryMode queryMode, typename InsideWallChecker>
__global__ void getBoundaryCells(real maximumTravel, CellListInfo cinfo,
                                 int *nBoundaryCells, int *boundaryCells,
                                 InsideWallChecker checker)
{
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= cinfo.totcells) return;

    const int3 ind = cinfo.decode(cid);
    const real3 cornerCoo = -0.5_r * cinfo.localDomainSize + make_real3(ind)*cinfo.h;

    if (isCellOnBoundary(maximumTravel, cornerCoo, cinfo.h, checker))
    {
        const int id = atomicAggInc(nBoundaryCells);
        if (queryMode == QueryMode::Collect)
            boundaryCells[id] = cid;
    }
}

//===============================================================================================
// Checking kernel
//===============================================================================================

template<typename InsideWallChecker>
__global__ void checkInside(PVview view, int *nInside, const InsideWallChecker checker)
{
    const real checkTolerance = 1e-4_r;

    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    const Real3_int coo(view.readPosition(pid));

    const real v = checker(coo.v);

    if (v > checkTolerance)
        atomicAggInc(nInside);
}

//===============================================================================================
// Kernels computing sdf and sdf gradient per particle
//===============================================================================================

template<typename InsideWallChecker>
__global__ void computeSdfPerParticle(PVview view, real gradientThreshold,
                                      real *sdfs, real3 *gradients,
                                      InsideWallChecker checker)
{
    constexpr real h = 0.25_r;
    constexpr real zeroTolerance = 1e-6_r;

    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    const auto r = make_real3(view.readPosition(pid));

    const real sdf = checker(r);
    sdfs[pid] = sdf;

    if (gradients != nullptr)
    {
        if (sdf > -gradientThreshold)
        {
            const real3 grad = computeGradient(checker, r, h);

            if (dot(grad, grad) < zeroTolerance)
                gradients[pid] = make_real3(0, 0, 0);
            else
                gradients[pid] = normalize(grad);
        }
        else
        {
            gradients[pid] = make_real3(0, 0, 0);
        }
    }
}


template<typename InsideWallChecker>
__global__ void computeSdfPerPosition(int n, const real3 *positions, real *sdfs, InsideWallChecker checker)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n) return;

    auto r = positions[pid];

    sdfs[pid] = checker(r);
}

template<typename InsideWallChecker>
__global__ void computeSdfOnGrid(CellListInfo gridInfo, real *sdfs, InsideWallChecker checker)
{
    const int nid = blockIdx.x * blockDim.x + threadIdx.x;

    if (nid >= gridInfo.totcells) return;

    const int3 cid3 = gridInfo.decode(nid);
    const real3 r = gridInfo.h * make_real3(cid3) + 0.5_r * gridInfo.h - 0.5*gridInfo.localDomainSize;

    sdfs[nid] = checker(r);
}

} // namespace stationary_walls_kernels

//===============================================================================================
// Member functions
//===============================================================================================

template<class InsideWallChecker>
SimpleStationaryWall<InsideWallChecker>::SimpleStationaryWall(const MirState *state, const std::string& name, InsideWallChecker&& insideWallChecker) :
    SDFBasedWall(state, name),
    insideWallChecker_(std::move(insideWallChecker))
{
    bounceForce_.clear(defaultStream);
}

template<class InsideWallChecker>
SimpleStationaryWall<InsideWallChecker>::~SimpleStationaryWall() = default;

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::setup(MPI_Comm& comm)
{
    info("Setting up wall %s", getCName());

    CUDA_Check( cudaDeviceSynchronize() );

    insideWallChecker_.setup(comm, getState()->domain);

    CUDA_Check( cudaDeviceSynchronize() );
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::setPrerequisites(ParticleVector *pv)
{
    // do not set it to persistent because bounce happens after integration
    pv->requireDataPerParticle<real4> (channel_names::oldPositions, DataManager::PersistenceMode::None, DataManager::ShiftMode::Active);
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::attachFrozen(ParticleVector *pv)
{
    frozen_ = pv;
    info("Wall '%s' will treat particle vector '%s' as frozen", getCName(), pv->getCName());
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::attach(ParticleVector *pv, CellList *cl, real maximumPartTravel)
{
    if (pv == frozen_)
    {
        info("Particle Vector '%s' declared as frozen for the wall '%s'. Bounce-back won't work",
             pv->getCName(), getCName());
        return;
    }

    if (dynamic_cast<PrimaryCellList*>(cl) == nullptr)
        die("PVs should only be attached to walls with the primary cell-lists! "
            "Invalid combination: wall %s, pv %s", getCName(), pv->getCName());

    CUDA_Check( cudaDeviceSynchronize() );
    particleVectors_.push_back(pv);
    cellLists_.push_back(cl);

    const int nthreads = 128;
    const int nblocks = getNblocks(cl->totcells, nthreads);

    PinnedBuffer<int> nBoundaryCells(1);
    nBoundaryCells.clear(defaultStream);

    SAFE_KERNEL_LAUNCH(
        stationary_walls_kernels::getBoundaryCells<QueryMode::Query>,
        nblocks, nthreads, 0, defaultStream,
        maximumPartTravel, cl->cellInfo(), nBoundaryCells.devPtr(),
        nullptr, insideWallChecker_.handler() );

    nBoundaryCells.downloadFromDevice(defaultStream);

    debug("Found %d boundary cells", nBoundaryCells[0]);
    DeviceBuffer<int> bc(nBoundaryCells[0]);

    nBoundaryCells.clear(defaultStream);
    SAFE_KERNEL_LAUNCH(
        stationary_walls_kernels::getBoundaryCells<QueryMode::Collect>,
        nblocks, nthreads, 0, defaultStream,
        maximumPartTravel, cl->cellInfo(), nBoundaryCells.devPtr(),
        bc.devPtr(), insideWallChecker_.handler() );

    boundaryCells_.push_back(std::move(bc));
    CUDA_Check( cudaDeviceSynchronize() );
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::detachAllCellLists()
{
    assert(particleVectors_.size() == cellLists_.size());
    assert(particleVectors_.size() == boundaryCells_.size());
    particleVectors_.clear();
    cellLists_.clear();
    boundaryCells_.clear();
}

static bool keepAllpersistentDataPredicate(const DataManager::NamedChannelDesc& namedDesc)
{
    return namedDesc.second->persistence == DataManager::PersistenceMode::Active;
};

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::removeInner(ParticleVector *pv)
{
    if (pv == frozen_)
    {
        warn("Particle Vector '%s' declared as frozen for the wall '%s'. Will not remove any particles from there",
             pv->getCName(), getCName());
        return;
    }

    CUDA_Check( cudaDeviceSynchronize() );

    PinnedBuffer<int> nRemaining(1);
    nRemaining.clear(defaultStream);

    const int oldSize = pv->local()->size();
    if (oldSize == 0) return;

    constexpr int nthreads = 128;
    // Need a different path for objects
    if (auto ov = dynamic_cast<ObjectVector*>(pv))
    {
        // Prepare temp storage for extra object data
        OVview ovView(ov, ov->local());
        ObjectPacker packer(keepAllpersistentDataPredicate);
        packer.update(ov->local(), defaultStream);
        const int maxNumObj = ovView.nObjects;

        DeviceBuffer<char> tmp(packer.getSizeBytes(maxNumObj));

        constexpr int warpSize = 32;

        SAFE_KERNEL_LAUNCH(
            stationary_walls_kernels::packRemainingObjects,
            getNblocks(ovView.nObjects*warpSize, nthreads), nthreads, 0, defaultStream,
            ovView, packer.handler(), tmp.devPtr(), nRemaining.devPtr(),
            insideWallChecker_.handler(), maxNumObj );

        nRemaining.downloadFromDevice(defaultStream);

        if (nRemaining[0] != ovView.nObjects)
        {
            info("Removing %d out of %d '%s' objects from walls '%s'",
                 ovView.nObjects - nRemaining[0], ovView.nObjects,
                 ov->getCName(), this->getCName());

            // Copy temporary buffers back
            ov->local()->resize_anew(nRemaining[0] * ov->getObjectSize());
            ovView = OVview(ov, ov->local());
            packer.update(ov->local(), defaultStream);

            SAFE_KERNEL_LAUNCH(
                stationary_walls_kernels::unpackRemainingObjects,
                ovView.nObjects, nthreads, 0, defaultStream,
                tmp.devPtr(), ovView, packer.handler(), maxNumObj );
        }
    }
    else
    {
        PVview view(pv, pv->local());
        ParticlePacker packer(keepAllpersistentDataPredicate);
        packer.update(pv->local(), defaultStream);
        const int maxNumParticles = view.size;

        DeviceBuffer<char> tmpBuffer(packer.getSizeBytes(maxNumParticles));

        SAFE_KERNEL_LAUNCH(
            stationary_walls_kernels::packRemainingParticles,
            getNblocks(view.size, nthreads), nthreads, 0, defaultStream,
            view, packer.handler(), tmpBuffer.devPtr(), nRemaining.devPtr(),
            insideWallChecker_.handler(), maxNumParticles );

        nRemaining.downloadFromDevice(defaultStream);
        const int newSize = nRemaining[0];

        if (newSize != oldSize)
        {
            info("Removing %d out of %d '%s' particles from walls '%s'",
                 oldSize - newSize, oldSize,
                 pv->getCName(), this->getCName());

            pv->local()->resize_anew(newSize);
            packer.update(pv->local(), defaultStream);

            SAFE_KERNEL_LAUNCH(
                stationary_walls_kernels::unpackRemainingParticles,
                getNblocks(newSize, nthreads), nthreads, 0, defaultStream,
                tmpBuffer.devPtr(), packer.handler(), newSize, maxNumParticles );
        }
    }

    pv->haloValid   = false;
    pv->redistValid = false;
    pv->cellListStamp++;

    info("Wall '%s' has removed inner entities of pv '%s', keeping %d out of %d particles",
         getCName(), pv->getCName(), pv->local()->size(), oldSize);

    CUDA_Check( cudaDeviceSynchronize() );
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::bounce(cudaStream_t stream)
{
    const real dt = this->getState()->getDt();

    bounceForce_.clear(stream);

    for (size_t i = 0; i < particleVectors_.size(); ++i)
    {
        auto  pv = particleVectors_[i];
        auto  cl = cellLists_[i];
        auto& bc = boundaryCells_[i];
        auto  view = cl->getView<PVviewWithOldParticles>();

        debug2("Bouncing %d %s particles, %zu boundary cells",
               pv->local()->size(), pv->getCName(), bc.size());

        const int nthreads = 64;
        SAFE_KERNEL_LAUNCH(
                bounce_kernels::sdfBounce,
                getNblocks(bc.size(), nthreads), nthreads, 0, stream,
                view, cl->cellInfo(),
                bc.devPtr(), bc.size(), dt,
                insideWallChecker_.handler(),
                VelocityFieldNone{},
                bounceForce_.devPtr());

        CUDA_Check( cudaPeekAtLastError() );
    }
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::check(cudaStream_t stream)
{
    constexpr int nthreads = 128;
    for (auto pv : particleVectors_)
    {
        nInside_.clearDevice(stream);
        const PVview view(pv, pv->local());

        SAFE_KERNEL_LAUNCH(
            stationary_walls_kernels::checkInside,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, nInside_.devPtr(), insideWallChecker_.handler() );

        nInside_.downloadFromDevice(stream);

        info("%d particles of %s are inside the wall %s", nInside_[0], pv->getCName(), getCName());
    }
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::sdfPerParticle(LocalParticleVector *lpv,
        GPUcontainer *sdfs, GPUcontainer *gradients, real gradientThreshold, cudaStream_t stream)
{
    const int nthreads = 128;
    const int np = lpv->size();
    auto pv = lpv->parent();

    if (sizeof(real) % sdfs->datatype_size() != 0)
        die("Incompatible datatype size of container for SDF values: %zu (working with PV '%s')",
            sdfs->datatype_size(), pv->getCName());
    sdfs->resize_anew( np * sizeof(real) / sdfs->datatype_size());


    if (gradients != nullptr)
    {
        if (sizeof(real3) % gradients->datatype_size() != 0)
            die("Incompatible datatype size of container for SDF gradients: %zu (working with PV '%s')",
                gradients->datatype_size(), pv->getCName());
        gradients->resize_anew( np * sizeof(real3) / gradients->datatype_size());
    }

    PVview view(pv, lpv);
    SAFE_KERNEL_LAUNCH(
        stationary_walls_kernels::computeSdfPerParticle,
        getNblocks(view.size, nthreads), nthreads, 0, stream,
        view, gradientThreshold, (real*)sdfs->genericDevPtr(),
        (gradients != nullptr) ? (real3*)gradients->genericDevPtr() : nullptr, insideWallChecker_.handler() );
}


template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::sdfPerPosition(GPUcontainer *positions, GPUcontainer *sdfs, cudaStream_t stream)
{
    const int n = positions->size();

    if (sizeof(real) % sdfs->datatype_size() != 0)
        die("Incompatible datatype size of container for SDF values: %zu (sampling sdf on positions)",
            sdfs->datatype_size());

    if (sizeof(real3) % sdfs->datatype_size() != 0)
        die("Incompatible datatype size of container for Psitions values: %zu (sampling sdf on positions)",
            positions->datatype_size());

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
        stationary_walls_kernels::computeSdfPerPosition,
        getNblocks(n, nthreads), nthreads, 0, stream,
        n, (real3*)positions->genericDevPtr(), (real*)sdfs->genericDevPtr(), insideWallChecker_.handler() );
}


template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::sdfOnGrid(real3 h, GPUcontainer *sdfs, cudaStream_t stream)
{
    if (sizeof(real) % sdfs->datatype_size() != 0)
        die("Incompatible datatype size of container for SDF values: %zu (sampling sdf on a grid)",
            sdfs->datatype_size());

    const CellListInfo gridInfo(h, getState()->domain.localSize);
    sdfs->resize_anew(gridInfo.totcells);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
        stationary_walls_kernels::computeSdfOnGrid,
        getNblocks(gridInfo.totcells, nthreads), nthreads, 0, stream,
        gridInfo, (real*) sdfs->genericDevPtr(), insideWallChecker_.handler() );
}

template<class InsideWallChecker>
PinnedBuffer<double3>* SimpleStationaryWall<InsideWallChecker>::getCurrentBounceForce()
{
    return &bounceForce_;
}

template class SimpleStationaryWall<StationaryWallSphere>;
template class SimpleStationaryWall<StationaryWallCylinder>;
template class SimpleStationaryWall<StationaryWallSDF>;
template class SimpleStationaryWall<StationaryWallPlane>;
template class SimpleStationaryWall<StationaryWallBox>;

} // namespace mirheo
