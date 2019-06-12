#include "simple_stationary_wall.h"

#include "common_kernels.h"
#include "stationary_walls/box.h"
#include "stationary_walls/cylinder.h"
#include "stationary_walls/plane.h"
#include "stationary_walls/sdf.h"
#include "stationary_walls/sphere.h"
#include "velocity_field/none.h"

#include <core/bounce_solver.h>
#include <core/celllist.h>
#include <core/field/utils.h>
#include <core/logger.h>
#include <core/pvs/packers/packers.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/ov.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <texture_types.h>

//===============================================================================================
// Removing kernels
//===============================================================================================

template<typename InsideWallChecker>
__global__ void collectRemaining(PVview view, float4 *remainingPos, float4 *remainingVel, int *nRemaining, InsideWallChecker checker)
{
    const float tolerance = 1e-6f;

    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Particle p(view.readParticle(pid));

    const float val = checker(p.r);

    if (val <= -tolerance)
    {
        const int ind = atomicAggInc(nRemaining);
        p.write2Float4(remainingPos, remainingVel, ind);
    }
}

template<typename InsideWallChecker>
__global__ void packRemainingObjects(OVview view, ObjectPacker packer, char *output, int *nRemaining, InsideWallChecker checker)
{
    const float tolerance = 1e-6f;

    // One warp per object
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int objId = gid / warpSize;
    const int tid = gid % warpSize;

    if (objId >= view.nObjects) return;

    bool isRemaining = true;
    for (int i = tid; i < view.objSize; i += warpSize)
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
    if (tid == 0)
        dstObjId = atomicAdd(nRemaining, 1);
    dstObjId = warpShfl(dstObjId, 0);

    char* dstAddr = output + dstObjId * packer.totalPackedSize_byte;
    for (int pid = tid; pid < view.objSize; pid += warpSize)
    {
        const int srcPid = objId * view.objSize + pid;
        packer.part.pack(srcPid, dstAddr + pid*packer.part.packedSize_byte);
    }

    dstAddr += view.objSize * packer.part.packedSize_byte;
    if (tid == 0) packer.obj.pack(objId, dstAddr);
}

__global__ static void unpackRemainingObjects(const char *from, OVview view, ObjectPacker packer)
{
    const int objId = blockIdx.x;
    const int tid = threadIdx.x;

    const char* srcAddr = from + packer.totalPackedSize_byte * objId;

    for (int pid = tid; pid < view.objSize; pid += blockDim.x)
    {
        const int dstId = objId*view.objSize + pid;
        packer.part.unpack(srcAddr + pid*packer.part.packedSize_byte, dstId);
    }

    srcAddr += view.objSize * packer.part.packedSize_byte;
    if (tid == 0) packer.obj.unpack(srcAddr, objId);
}
//===============================================================================================
// Boundary cells kernels
//===============================================================================================

template<typename InsideWallChecker>
__device__ inline bool isCellOnBoundary(const float maximumTravel, float3 cornerCoo, float3 len, InsideWallChecker checker)
{
    int pos = 0, neg = 0;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
            {
                // Value in the cell corner
                const float3 shift = make_float3(i ? len.x : 0.0f, j ? len.y : 0.0f, k ? len.z : 0.0f);
                const float s = checker(cornerCoo + shift);

                if (s >  maximumTravel) pos++;
                if (s < -maximumTravel) neg++;
            }

    return (pos != 8 && neg != 8);
}

enum class QueryMode {
   Query,
   Collect    
};

template<QueryMode queryMode, typename InsideWallChecker>
__global__ void getBoundaryCells(float maximumTravel, CellListInfo cinfo, int *nBoundaryCells, int *boundaryCells, InsideWallChecker checker)
{
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= cinfo.totcells) return;

    int3 ind;
    cinfo.decode(cid, ind.x, ind.y, ind.z);
    float3 cornerCoo = -0.5f*cinfo.localDomainSize + make_float3(ind)*cinfo.h;

    if (isCellOnBoundary(maximumTravel, cornerCoo, cinfo.h, checker))
    {
        int id = atomicAggInc(nBoundaryCells);
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
	const float checkTolerance = 1e-4f;

    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Float3_int coo(view.readPosition(pid));

    float v = checker(coo.v);

    if (v > checkTolerance) atomicAggInc(nInside);
}

//===============================================================================================
// Kernels computing sdf and sdf gradient per particle
//===============================================================================================

template<typename InsideWallChecker>
__global__ void computeSdfPerParticle(PVview view, float gradientThreshold, float *sdfs, float3 *gradients, InsideWallChecker checker)
{
    const float h = 0.25f;
    const float zeroTolerance = 1e-10f;

    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Particle p;
    view.readPosition(p, pid);

    float sdf = checker(p.r);
    sdfs[pid] = sdf;

    if (gradients != nullptr && sdf > -gradientThreshold)
    {
        float3 grad = computeGradient(checker, p.r, h);

        if (dot(grad, grad) < zeroTolerance)
            gradients[pid] = make_float3(0, 0, 0);
        else
            gradients[pid] = normalize(grad);
    }
}


template<typename InsideWallChecker>
__global__ void computeSdfPerPosition(int n, const float3 *positions, float *sdfs, InsideWallChecker checker)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n) return;
    
    auto r = positions[pid];    

    sdfs[pid] = checker(r);
}

template<typename InsideWallChecker>
__global__ void computeSdfOnGrid(CellListInfo gridInfo, float *sdfs, InsideWallChecker checker)
{
    const int nid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (nid >= gridInfo.totcells) return;
    
    const int3 cid3 = gridInfo.decode(nid);
    const float3 r = gridInfo.h * make_float3(cid3) + 0.5f*gridInfo.h - 0.5*gridInfo.localDomainSize;
    
    sdfs[nid] = checker(r);
}

//===============================================================================================
// Member functions
//===============================================================================================

template<class InsideWallChecker>
SimpleStationaryWall<InsideWallChecker>::SimpleStationaryWall(std::string name, const YmrState *state, InsideWallChecker&& insideWallChecker) :
    SDF_basedWall(state, name),
    insideWallChecker(std::move(insideWallChecker))
{
    bounceForce.clear(0);
}

template<class InsideWallChecker>
SimpleStationaryWall<InsideWallChecker>::~SimpleStationaryWall() = default;

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::setup(MPI_Comm& comm)
{
    info("Setting up wall %s", name.c_str());

    CUDA_Check( cudaDeviceSynchronize() );

    insideWallChecker.setup(comm, state->domain);

    CUDA_Check( cudaDeviceSynchronize() );
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::attachFrozen(ParticleVector *pv)
{
    frozen = pv;
    info("Wall '%s' will treat particle vector '%s' as frozen", name.c_str(), pv->name.c_str());
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::attach(ParticleVector *pv, CellList *cl, float maximumPartTravel)
{
    if (pv == frozen)
    {
        warn("Particle Vector '%s' declared as frozen for the wall '%s'. Bounce-back won't work",
             pv->name.c_str(), name.c_str());
        return;
    }
    
    if (dynamic_cast<PrimaryCellList*>(cl) == nullptr)
        die("PVs should only be attached to walls with the primary cell-lists! "
            "Invalid combination: wall %s, pv %s", name.c_str(), pv->name.c_str());

    CUDA_Check( cudaDeviceSynchronize() );
    particleVectors.push_back(pv);
    cellLists.push_back(cl);

    const int nthreads = 128;
    const int nblocks = getNblocks(cl->totcells, nthreads);
    
    PinnedBuffer<int> nBoundaryCells(1);
    nBoundaryCells.clear(defaultStream);

    SAFE_KERNEL_LAUNCH(
            getBoundaryCells<QueryMode::Query>,
            nblocks, nthreads, 0, defaultStream,
            maximumPartTravel, cl->cellInfo(), nBoundaryCells.devPtr(),
            nullptr, insideWallChecker.handler() );

    nBoundaryCells.downloadFromDevice(defaultStream);

    debug("Found %d boundary cells", nBoundaryCells[0]);
    DeviceBuffer<int> bc(nBoundaryCells[0]);

    nBoundaryCells.clear(defaultStream);
    SAFE_KERNEL_LAUNCH(
            getBoundaryCells<QueryMode::Collect>,
            nblocks, nthreads, 0, defaultStream,
            maximumPartTravel, cl->cellInfo(), nBoundaryCells.devPtr(),
            bc.devPtr(), insideWallChecker.handler() );

    boundaryCells.push_back(std::move(bc));
    CUDA_Check( cudaDeviceSynchronize() );
}



template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::removeInner(ParticleVector *pv)
{
    if (pv == frozen)
    {
        warn("Particle Vector '%s' declared as frozen for the wall '%s'. Will not remove any particles from there",
             pv->name.c_str(), name.c_str());
        return;
    }
    
    CUDA_Check( cudaDeviceSynchronize() );

    PinnedBuffer<int> nRemaining(1);
    nRemaining.clear(defaultStream);

    int oldSize = pv->local()->size();
    if (oldSize == 0) return;

    const int nthreads = 128;
    // Need a different path for objects
    ObjectVector* ov = dynamic_cast<ObjectVector*>(pv);
    if (ov == nullptr)
    {
        PVview view(pv, pv->local());
        PinnedBuffer<float4> tmpPos(view.size), tmpVel(view.size);

        SAFE_KERNEL_LAUNCH(
                collectRemaining,
                getNblocks(view.size, nthreads), nthreads, 0, defaultStream,
                view, tmpPos.devPtr(), tmpVel.devPtr(), nRemaining.devPtr(),
                insideWallChecker.handler() );

        nRemaining.downloadFromDevice(defaultStream);
        std::swap(pv->local()->positions(),  tmpPos);
        std::swap(pv->local()->velocities(), tmpVel);
        pv->local()->resize(nRemaining[0], defaultStream);
    }
    else
    {
        PackPredicate packPredicate = [](const DataManager::NamedChannelDesc& namedDesc) {
            return namedDesc.second->persistence == DataManager::PersistenceMode::Persistent;
        };
        
        // Prepare temp storage for extra object data
        OVview ovView(ov, ov->local());
        ObjectPacker packer(ov, ov->local(), packPredicate, defaultStream);

        DeviceBuffer<char> tmp(ovView.nObjects * packer.totalPackedSize_byte);

        SAFE_KERNEL_LAUNCH(
                packRemainingObjects,
                getNblocks(ovView.nObjects*32, nthreads), nthreads, 0, defaultStream,
                ovView,    packer, tmp.devPtr(), nRemaining.devPtr(), insideWallChecker.handler() );

        // Copy temporary buffers back
        nRemaining.downloadFromDevice(defaultStream);
        ov->local()->resize_anew(nRemaining[0] * ov->objSize);
        ovView = OVview(ov, ov->local());
        packer = ObjectPacker(ov, ov->local(), packPredicate, defaultStream);

        SAFE_KERNEL_LAUNCH(
                unpackRemainingObjects,
                ovView.nObjects, nthreads, 0, defaultStream,
                tmp.devPtr(), ovView, packer  );
    }

    pv->haloValid = false;
    pv->redistValid = false;
    pv->cellListStamp++;

    info("Wall '%s' has removed inner entities of pv '%s', keeping %d out of %d particles",
         name.c_str(), pv->name.c_str(), pv->local()->size(), oldSize);

    CUDA_Check( cudaDeviceSynchronize() );
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::bounce(cudaStream_t stream)
{
    float dt = this->state->dt;

    bounceForce.clear(stream);
    
    for (int i = 0; i < particleVectors.size(); i++)
    {
        auto  pv = particleVectors[i];
        auto  cl = cellLists[i];
        auto& bc = boundaryCells[i];
        auto  view = cl->getView<PVviewWithOldParticles>();

        debug2("Bouncing %d %s particles, %d boundary cells",
               pv->local()->size(), pv->name.c_str(), bc.size());

        const int nthreads = 64;
        SAFE_KERNEL_LAUNCH(
                BounceKernels::sdfBounce,
                getNblocks(bc.size(), nthreads), nthreads, 0, stream,
                view, cl->cellInfo(),
                bc.devPtr(), bc.size(), dt,
                insideWallChecker.handler(),
                VelocityField_None(),
                bounceForce.devPtr());

        CUDA_Check( cudaPeekAtLastError() );
    }
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::check(cudaStream_t stream)
{
    const int nthreads = 128;
    for (int i=0; i<particleVectors.size(); i++)
    {
        auto pv = particleVectors[i];
        {
            nInside.clearDevice(stream);
            PVview view(pv, pv->local());
            SAFE_KERNEL_LAUNCH(
                    checkInside,
                    getNblocks(view.size, nthreads), nthreads, 0, stream,
                    view, nInside.devPtr(), insideWallChecker.handler() );

            nInside.downloadFromDevice(stream);

            say("%d particles of %s are inside the wall %s", nInside[0], pv->name.c_str(), name.c_str());
        }
    }
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::sdfPerParticle(LocalParticleVector* lpv,
        GPUcontainer* sdfs, GPUcontainer* gradients, float gradientThreshold, cudaStream_t stream)
{
    const int nthreads = 128;
    const int np = lpv->size();
    auto pv = lpv->pv;

    if (sizeof(float) % sdfs->datatype_size() != 0)
        die("Incompatible datatype size of container for SDF values: %d (working with PV '%s')",
            sdfs->datatype_size(), pv->name.c_str());
    sdfs->resize_anew( np*sizeof(float) / sdfs->datatype_size());

    
    if (gradients != nullptr)
    {
        if (sizeof(float3) % gradients->datatype_size() != 0)
            die("Incompatible datatype size of container for SDF gradients: %d (working with PV '%s')",
                gradients->datatype_size(), pv->name.c_str());
        gradients->resize_anew( np*sizeof(float3) / gradients->datatype_size());
    }

    PVview view(pv, lpv);
    SAFE_KERNEL_LAUNCH(
            computeSdfPerParticle,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, gradientThreshold, (float*)sdfs->genericDevPtr(),
            (gradients != nullptr) ? (float3*)gradients->genericDevPtr() : nullptr, insideWallChecker.handler() );
}


template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::sdfPerPosition(GPUcontainer *positions, GPUcontainer* sdfs, cudaStream_t stream)
{
    int n = positions->size();
    
    if (sizeof(float) % sdfs->datatype_size() != 0)
        die("Incompatible datatype size of container for SDF values: %d (sampling sdf on positions)",
            sdfs->datatype_size());

    if (sizeof(float3) % sdfs->datatype_size() != 0)
        die("Incompatible datatype size of container for Psitions values: %d (sampling sdf on positions)",
            positions->datatype_size());
    
    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            computeSdfPerPosition,
            getNblocks(n, nthreads), nthreads, 0, stream,
            n, (float3*)positions->genericDevPtr(), (float*)sdfs->genericDevPtr(), insideWallChecker.handler() );
}


template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::sdfOnGrid(float3 h, GPUcontainer* sdfs, cudaStream_t stream)
{
    if (sizeof(float) % sdfs->datatype_size() != 0)
        die("Incompatible datatype size of container for SDF values: %d (sampling sdf on a grid)",
            sdfs->datatype_size());
        
    CellListInfo gridInfo(h, state->domain.localSize);
    sdfs->resize_anew(gridInfo.totcells);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            computeSdfOnGrid,
            getNblocks(gridInfo.totcells, nthreads), nthreads, 0, stream,
            gridInfo, (float*)sdfs->genericDevPtr(), insideWallChecker.handler() );
}

template<class InsideWallChecker>
PinnedBuffer<double3>* SimpleStationaryWall<InsideWallChecker>::getCurrentBounceForce()
{
    return &bounceForce;
}

template class SimpleStationaryWall<StationaryWall_Sphere>;
template class SimpleStationaryWall<StationaryWall_Cylinder>;
template class SimpleStationaryWall<StationaryWall_SDF>;
template class SimpleStationaryWall<StationaryWall_Plane>;
template class SimpleStationaryWall<StationaryWall_Box>;




