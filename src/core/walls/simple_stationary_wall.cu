#include "simple_stationary_wall.h"

#include <fstream>
#include <cmath>
#include <texture_types.h>
#include <cassert>

#include <core/logger.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/cuda_common.h>
#include <core/celllist.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/object_vector.h>
#include <core/pvs/views/ov.h>
#include <core/pvs/extra_data/packers.h>
#include <core/bounce_solver.h>

#include <core/utils/cuda_rng.h>

#include "stationary_walls/cylinder.h"
#include "stationary_walls/sdf.h"
#include "stationary_walls/sphere.h"
#include "stationary_walls/plane.h"
#include "stationary_walls/box.h"

//===============================================================================================
// Removing kernels
//===============================================================================================

template<typename InsideWallChecker>
__global__ void collectRemaining(PVview view, float4* remaining, int* nRemaining, InsideWallChecker checker)
{
    const float tolerance = 1e-6f;

    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Particle p(view.particles, pid);

    const float val = checker(p.r);

    if (val <= -tolerance)
    {
        const int ind = atomicAggInc(nRemaining);
        p.write2Float4(remaining, ind);
    }
}

template<typename InsideWallChecker>
__global__ void packRemainingObjects(OVview view, ObjectPacker packer, char* output, int* nRemaining, InsideWallChecker checker)
{
    const float tolerance = 1e-6f;

    // One warp per object
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int objId = gid / warpSize;
    const int tid = gid % warpSize;

    if (objId >= view.nObjects) return;

    bool isRemaining = true;
    for (int i=tid; i < view.objSize; i+=warpSize)
    {
        Particle p(view.particles, objId * view.objSize + i);
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

__global__ static void unpackRemainingObjects(const char* from, OVview view, ObjectPacker packer)
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
__device__ inline bool isCellOnBoundary(PVview view, float3 cornerCoo, float3 len, InsideWallChecker checker)
{
    // About maximum distance a particle can cover in one step
    const float tol = 0.25f;
    int pos = 0, neg = 0;

    for (int i=0; i<2; i++)
        for (int j=0; j<2; j++)
            for (int k=0; k<2; k++)
            {
                // Value in the cell corner
                const float3 shift = make_float3(i ? len.x : 0.0f, j ? len.y : 0.0f, k ? len.z : 0.0f);
                const float s = checker(cornerCoo + shift);

                if (s >  tol) pos++;
                if (s < -tol) neg++;
            }

    return (pos != 8 && neg != 8);
}

template<bool QUERY, typename InsideWallChecker>
__global__ void getBoundaryCells(PVview view, CellListInfo cinfo, int* nBoundaryCells, int* boundaryCells, InsideWallChecker checker)
{
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= cinfo.totcells) return;

    int3 ind;
    cinfo.decode(cid, ind.x, ind.y, ind.z);
    float3 cornerCoo = -0.5f*cinfo.localDomainSize + make_float3(ind)*cinfo.h;

    if (isCellOnBoundary(view, cornerCoo, cinfo.h, checker))
    {
        int id = atomicAggInc(nBoundaryCells);
        if (!QUERY) boundaryCells[id] = cid;
    }
}

//===============================================================================================
// SDF bouncing kernel
//===============================================================================================

template<typename InsideWallChecker>
__device__ float3 rescue(float3 candidate, float dt, float tol, int id, const InsideWallChecker& checker)
{
    const int maxIters = 100;
    const float factor = 5.0f*dt;
    
    for (int i=0; i<maxIters; i++)
    {
        float v = checker(candidate);
        if (v < -tol) break;
        
        float3 rndShift;
        rndShift.x = Saru::mean0var1(candidate.x - floorf(candidate.x), id+i, id*id);
        rndShift.y = Saru::mean0var1(rndShift.x,                        id+i, id*id);
        rndShift.z = Saru::mean0var1(rndShift.y,                        id+i, id*id);

        if (checker(candidate + factor*rndShift) < v)
            candidate += factor*rndShift;
    }

    return candidate;
}

template<typename InsideWallChecker>
__global__ void bounceKernel(
        PVviewWithOldParticles view, CellListInfo cinfo,
        const int* wallCells, const int nWallCells, const float dt, const InsideWallChecker checker)
{
    const float insideTolerance = 2e-6f;

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nWallCells) return;
    const int cid = wallCells[tid];
    const int pstart = cinfo.cellStarts[cid];
    const int pend   = cinfo.cellStarts[cid+1];

    for (int pid = pstart; pid < pend; pid++)
    {
        Particle p(view.particles, pid);
        
        const float val = checker(p.r);        
        if (val < 0.0f) continue;

        float3 candidate;
        Particle pOld(view.old_particles, pid);
        const float oldVal = checker(pOld.r);
        
        // If for whatever reason the previous position was bad, try to rescue
        if (oldVal >= 0.0f) candidate = rescue(pOld.r, dt, insideTolerance, p.i1, checker);
        
        // If the previous position was very close to the surface,
        // remain there and simply reverse the velocity
        else if (oldVal > -insideTolerance) candidate = pOld.r;
        else
        {
            // Otherwise go to the point where sdf = 2*insideTolerance
            float3 dr = p.r - pOld.r;

            const float2 alpha_val = solveLinSearch_verbose([=] (float lambda) {
                return checker(pOld.r + dr*lambda) + insideTolerance;
            });
            
            if (alpha_val.x >= 0.0f && alpha_val.y < 0.0f)
                candidate = pOld.r + dr*alpha_val.x;
            else
                candidate = pOld.r;
        }

        p.r = candidate;
        p.u = -p.u;

        p.write2Float4(cinfo.particles, pid);
    }
}

//===============================================================================================
// Checking kernel
//===============================================================================================

template<typename InsideWallChecker>
__global__ void checkInside(PVview view, int* nInside, const InsideWallChecker checker)
{
	const float checkTolerance = 1e-4f;

    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Float3_int coo(view.particles[2*pid]);

    float v = checker(coo.v);

    if (v > checkTolerance) atomicAggInc(nInside);
}

//===============================================================================================
// Kernels computing sdf and sdf gradient per particle
//===============================================================================================

template<typename InsideWallChecker>
__global__ void computeSdfPerParticle(PVview view, float* sdfs, float3* gradients, InsideWallChecker checker)
{
    const float h = 0.25f;

    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Particle p;
    p.readCoordinate(view.particles, pid);

    float sdf = checker(p.r);
    sdfs[pid] = sdf;

    if (gradients != nullptr)
    {
        float sdf_mx = checker(p.r + make_float3(-h,  0,  0));
        float sdf_px = checker(p.r + make_float3( h,  0,  0));
        float sdf_my = checker(p.r + make_float3( 0, -h,  0));
        float sdf_py = checker(p.r + make_float3( 0,  h,  0));
        float sdf_mz = checker(p.r + make_float3( 0,  0, -h));
        float sdf_pz = checker(p.r + make_float3( 0,  0,  h));

        float3 grad = make_float3( sdf_px - sdf_mx, sdf_py - sdf_my, sdf_pz - sdf_mz ) * (1.0f / (2.0f*h));

        gradients[pid] = normalize(grad);
    }
}


template<typename InsideWallChecker>
__global__ void computeSdfOnGrid(CellListInfo gridInfo, float* sdfs, InsideWallChecker checker)
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
void SimpleStationaryWall<InsideWallChecker>::setup(MPI_Comm& comm, DomainInfo domain)
{
    info("Setting up wall %s", name.c_str());

    CUDA_Check( cudaDeviceSynchronize() );
    MPI_Check( MPI_Comm_dup(comm, &wallComm) );

    insideWallChecker.setup(wallComm, domain);
    this->domain = domain;

    CUDA_Check( cudaDeviceSynchronize() );
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::attachFrozen(ParticleVector* pv)
{
    frozen = pv;
    info("Wall '%s' will treat particle vector '%s' as frozen", name.c_str(), pv->name.c_str());
}

template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::attach(ParticleVector* pv, CellList* cl)
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
    nBounceCalls.push_back(0);

    PVview view(pv, pv->local());
    PinnedBuffer<int> nBoundaryCells(1);
    nBoundaryCells.clear(0);
    SAFE_KERNEL_LAUNCH(
            getBoundaryCells<true>,
            (cl->totcells + 127) / 128, 128, 0, 0,
            view, cl->cellInfo(), nBoundaryCells.devPtr(), nullptr, insideWallChecker.handler() );

    nBoundaryCells.downloadFromDevice(0);

    debug("Found %d boundary cells", nBoundaryCells[0]);
    auto bc = new DeviceBuffer<int>(nBoundaryCells[0]);

    nBoundaryCells.clear(0);
    SAFE_KERNEL_LAUNCH(
            getBoundaryCells<false>,
            (cl->totcells + 127) / 128, 128, 0, 0,
            view, cl->cellInfo(), nBoundaryCells.devPtr(), bc->devPtr(), insideWallChecker.handler() );

    boundaryCells.push_back(bc);
    CUDA_Check( cudaDeviceSynchronize() );
}



template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::removeInner(ParticleVector* pv)
{
    if (pv == frozen)
    {
        warn("Particle Vector '%s' declared as frozen for the wall '%s'. Will not remove any particles from there",
            pv->name.c_str(), name.c_str());
        return;
    }
    
    CUDA_Check( cudaDeviceSynchronize() );

    PinnedBuffer<int> nRemaining(1);
    nRemaining.clear(0);

    int oldSize = pv->local()->size();
    if (oldSize == 0) return;

    const int nthreads = 128;
    // Need a different path for objects
    ObjectVector* ov = dynamic_cast<ObjectVector*>(pv);
    if (ov == nullptr)
    {
        PVview view(pv, pv->local());
        PinnedBuffer<Particle> tmp(view.size);

        SAFE_KERNEL_LAUNCH(
                collectRemaining,
                getNblocks(view.size, nthreads), nthreads, 0, 0,
                view, (float4*)tmp.devPtr(), nRemaining.devPtr(), insideWallChecker.handler() );

        nRemaining.downloadFromDevice(0);
        std::swap(pv->local()->coosvels, tmp);
        pv->local()->resize(nRemaining[0], 0);
    }
    else
    {
        // Prepare temp storage for extra object data
        OVview ovView(ov, ov->local());
        ObjectPacker packer(ov, ov->local(), 0);

        DeviceBuffer<char> tmp(ovView.nObjects * packer.totalPackedSize_byte);

        SAFE_KERNEL_LAUNCH(
                packRemainingObjects,
                getNblocks(ovView.nObjects*32, nthreads), nthreads, 0, 0,
                ovView,    packer, tmp.devPtr(), nRemaining.devPtr(), insideWallChecker.handler() );

        // Copy temporary buffers back
        nRemaining.downloadFromDevice(0);
        ov->local()->resize_anew(nRemaining[0] * ov->objSize);
        ovView = OVview(ov, ov->local());
        packer = ObjectPacker(ov, ov->local(), 0);

        SAFE_KERNEL_LAUNCH(
                unpackRemainingObjects,
                ovView.nObjects, nthreads, 0, 0,
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
void SimpleStationaryWall<InsideWallChecker>::bounce(float dt, cudaStream_t stream)
{
    for (int i=0; i<particleVectors.size(); i++)
    {
        auto pv = particleVectors[i];
        auto cl = cellLists[i];
        auto bc = boundaryCells[i];
        PVviewWithOldParticles view(pv, pv->local());

        debug2("Bouncing %d %s particles, %d boundary cells",
                pv->local()->size(), pv->name.c_str(), bc->size());

        const int nthreads = 64;
        SAFE_KERNEL_LAUNCH(
                bounceKernel,
                getNblocks(bc->size(), nthreads), nthreads, 0, stream,
                view, cl->cellInfo(), bc->devPtr(), bc->size(), dt, insideWallChecker.handler() );

        CUDA_Check( cudaPeekAtLastError() );
        nBounceCalls[i]++;
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
void SimpleStationaryWall<InsideWallChecker>::sdfPerParticle(LocalParticleVector* lpv, GPUcontainer* sdfs, GPUcontainer* gradients, cudaStream_t stream)
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
            view, (float*)sdfs->genericDevPtr(),
            (gradients != nullptr) ? (float3*)gradients->genericDevPtr() : nullptr, insideWallChecker.handler() );
}


template<class InsideWallChecker>
void SimpleStationaryWall<InsideWallChecker>::sdfOnGrid(float3 h, GPUcontainer* sdfs, cudaStream_t stream)
{
    if (sizeof(float) % sdfs->datatype_size() != 0)
        die("Incompatible datatype size of container for SDF values: %d (sampling sdf on a grid)",
            sdfs->datatype_size());
        
    CellListInfo gridInfo(h, domain.localSize);
    sdfs->resize_anew(gridInfo.totcells);

    const int nthreads = 128;
    SAFE_KERNEL_LAUNCH(
            computeSdfOnGrid,
            getNblocks(gridInfo.totcells, nthreads), nthreads, 0, stream,
            gridInfo, (float*)sdfs->genericDevPtr(), insideWallChecker.handler() );
}

template class SimpleStationaryWall<StationaryWall_Sphere>;
template class SimpleStationaryWall<StationaryWall_Cylinder>;
template class SimpleStationaryWall<StationaryWall_SDF>;
template class SimpleStationaryWall<StationaryWall_Plane>;
template class SimpleStationaryWall<StationaryWall_Box>;




