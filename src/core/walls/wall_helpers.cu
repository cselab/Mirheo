#include "wall_helpers.h"

#include <core/celllist.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>
#include <core/walls/simple_stationary_wall.h>
#include <core/xdmf/xdmf.h>

#include <curand_kernel.h>

namespace WallHelpersKernels
{
__global__ void init_sdf(int n, float *sdfs, float val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sdfs[i] = val;
}

__global__ void merge_sdfs(int n, const float *sdfs, float *sdfs_merged)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sdfs_merged[i] = max(sdfs[i], sdfs_merged[i]);
}

template<bool QUERY>
__global__ void collectFrozen(PVview view, const float *sdfs, float minVal, float maxVal,
                              float4 *frozenPos, float4 *frozenVel, int *nFrozen)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Particle p(view.readParticle(pid));
    p.u = make_float3(0);

    const float val = sdfs[pid];
        
    if (val > minVal && val < maxVal)
    {
        const int ind = atomicAggInc(nFrozen);

        if (!QUERY)
            p.write2Float4(frozenPos, frozenVel, ind);
    }
}

__global__ void initRandomPositions(int n, float3 *positions, long seed, float3 localSize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;
        
    curandState_t state;
    float3 r;
        
    curand_init(seed, i, 0, &state);
        
    r.x = localSize.x * (curand_uniform(&state) - 0.5f);
    r.y = localSize.y * (curand_uniform(&state) - 0.5f);
    r.z = localSize.z * (curand_uniform(&state) - 0.5f);

    positions[i] = r;
}

__global__ void countInside(int n, const float *sdf, int *nInside, float threshold = 0.f)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    int myval = 0;

    if (i < n)
        myval = sdf[i] < threshold;

    myval = warpReduce(myval, [] (int a, int b) {return a + b;});

    if (__laneid() == 0)
        atomicAdd(nInside, myval);
}
} // namespace WallHelpersKernels

static void extract_particles(ParticleVector *pv, const float *sdfs, float minVal, float maxVal)
{
    PinnedBuffer<int> nFrozen(1);
    
    PVview view(pv, pv->local());
    const int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    nFrozen.clear(defaultStream);

    SAFE_KERNEL_LAUNCH(
        WallHelpersKernels::collectFrozen<true>,
        nblocks, nthreads, 0, defaultStream,
        view, sdfs, minVal, maxVal, nullptr, nullptr, nFrozen.devPtr());

    nFrozen.downloadFromDevice(defaultStream);

    PinnedBuffer<float4> frozenPos(nFrozen[0]), frozenVel(nFrozen[0]);
    info("Freezing %d particles", nFrozen[0]);

    pv->local()->resize(nFrozen[0], defaultStream);

    nFrozen.clear(defaultStream);
    
    SAFE_KERNEL_LAUNCH(
        WallHelpersKernels::collectFrozen<false>,
        nblocks, nthreads, 0, defaultStream,
        view, sdfs, minVal, maxVal, frozenPos.devPtr(), frozenVel.devPtr(), nFrozen.devPtr());

    CUDA_Check( cudaDeviceSynchronize() );
    std::swap(frozenPos, pv->local()->positions());
    std::swap(frozenVel, pv->local()->velocities());
}

void WallHelpers::freezeParticlesInWall(SDF_basedWall *wall, ParticleVector *pv, float minVal, float maxVal)
{
    CUDA_Check( cudaDeviceSynchronize() );

    DeviceBuffer<float> sdfs(pv->local()->size());
    
    wall->sdfPerParticle(pv->local(), &sdfs, nullptr, 0, defaultStream);

    extract_particles(pv, sdfs.devPtr(), minVal, maxVal);
}


void WallHelpers::freezeParticlesInWalls(std::vector<SDF_basedWall*> walls, ParticleVector *pv, float minVal, float maxVal)
{
    CUDA_Check( cudaDeviceSynchronize() );

    int n = pv->local()->size();
    DeviceBuffer<float> sdfs(n), sdfs_merged(n);

    const int nthreads = 128;
    const int nblocks = getNblocks(n, nthreads);
    const float safety = 1.f;

    SAFE_KERNEL_LAUNCH(
        WallHelpersKernels::init_sdf,
        nblocks, nthreads, 0, defaultStream,
        n, sdfs_merged.devPtr(), minVal - safety);
    
    
    for (auto& wall : walls) {
        wall->sdfPerParticle(pv->local(), &sdfs, nullptr, 0, defaultStream);

        SAFE_KERNEL_LAUNCH(
            WallHelpersKernels::merge_sdfs,
            nblocks, nthreads, 0, defaultStream,
            n, sdfs.devPtr(), sdfs_merged.devPtr());
    }

    extract_particles(pv, sdfs_merged.devPtr(), minVal, maxVal);
}


void WallHelpers::dumpWalls2XDMF(std::vector<SDF_basedWall*> walls, float3 gridH, DomainInfo domain, std::string filename, MPI_Comm cartComm)
{
    CUDA_Check( cudaDeviceSynchronize() );
    CellListInfo gridInfo(gridH, domain.localSize);
    const int n = gridInfo.totcells;

    DeviceBuffer<float> sdfs(n);
    PinnedBuffer<float> sdfs_merged(n);
        
    const int nthreads = 128;
    const int nblocks = getNblocks(n, nthreads);
    const float initial = -1e5;

    SAFE_KERNEL_LAUNCH(
        WallHelpersKernels::init_sdf,
        nblocks, nthreads, 0, defaultStream,
        n, sdfs_merged.devPtr(), initial);
    
    for (auto& wall : walls)
    {
        wall->sdfOnGrid(gridH, &sdfs, 0);

        SAFE_KERNEL_LAUNCH(
            WallHelpersKernels::merge_sdfs,
            nblocks, nthreads, 0, defaultStream,
            n, sdfs.devPtr(), sdfs_merged.devPtr());
    }

    sdfs_merged.downloadFromDevice(defaultStream);
    
    XDMF::UniformGrid grid(gridInfo.ncells, gridInfo.h, cartComm);
    XDMF::Channel sdfCh("sdf", (void*)sdfs_merged.hostPtr(), XDMF::Channel::DataForm::Scalar, XDMF::Channel::NumberType::Float, DataTypeWrapper<float>());
    XDMF::write(filename, &grid, std::vector<XDMF::Channel>{sdfCh}, cartComm);
}


double WallHelpers::volumeInsideWalls(std::vector<SDF_basedWall*> walls, DomainInfo domain, MPI_Comm comm, long nSamplesPerRank)
{
    long n = nSamplesPerRank;
    DeviceBuffer<float3> positions(n);
    DeviceBuffer<float> sdfs(n), sdfs_merged(n);
    PinnedBuffer<int> nInside(1);

    const int nthreads = 128;
    const int nblocks = getNblocks(n, nthreads);
    const float initial = -1e5;

    SAFE_KERNEL_LAUNCH(
        WallHelpersKernels::initRandomPositions,
        nblocks, nthreads, 0, defaultStream,
        n, positions.devPtr(), 424242, domain.localSize);

    SAFE_KERNEL_LAUNCH(
        WallHelpersKernels::init_sdf,
        nblocks, nthreads, 0, defaultStream,
        n, sdfs_merged.devPtr(), initial);
        
    for (auto& wall : walls) {
        wall->sdfPerPosition(&positions, &sdfs, defaultStream);

        SAFE_KERNEL_LAUNCH(
            WallHelpersKernels::merge_sdfs,
            nblocks, nthreads, 0, defaultStream,
            n, sdfs.devPtr(), sdfs_merged.devPtr());
    }

    nInside.clear(defaultStream);
    
    SAFE_KERNEL_LAUNCH(
        WallHelpersKernels::countInside,
        nblocks, nthreads, 0, defaultStream,
        n, sdfs_merged.devPtr(), nInside.devPtr());

    nInside.downloadFromDevice(defaultStream, ContainersSynch::Synch);

    float3 localSize = domain.localSize;
    double subDomainVolume = localSize.x * localSize.y * localSize.z;

    double locVolume = (double) nInside[0] / (double) n * subDomainVolume;
    double totVolume = 0;

    MPI_Check( MPI_Allreduce(&locVolume, &totVolume, 1, MPI_DOUBLE, MPI_SUM, comm) );
    
    return totVolume;
}
