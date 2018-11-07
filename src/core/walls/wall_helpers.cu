#include <curand_kernel.h>

#include "wall_helpers.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cuda_common.h>
#include <core/celllist.h>
#include <core/utils/kernel_launch.h>
#include <core/xdmf/xdmf.h>

#include <core/walls/simple_stationary_wall.h>

static const cudaStream_t default_stream = 0;

namespace wall_helpers_kernels
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
    __global__ void collectFrozen(PVview view, const float *sdfs, float minVal, float maxVal, float4* frozen, int* nFrozen)
    {
        const int pid = blockIdx.x * blockDim.x + threadIdx.x;
        if (pid >= view.size) return;

        Particle p(view.particles, pid);
        p.u = make_float3(0);

        const float val = sdfs[pid];
        
        if (val > minVal && val < maxVal)
        {
            const int ind = atomicAggInc(nFrozen);

            if (!QUERY)
                p.write2Float4(frozen, ind);
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

        if (threadIdx.x % warpSize == 0)
            atomicAdd(nInside, myval);
    }
}

static void extract_particles(ParticleVector *pv, const float *sdfs, float minVal, float maxVal)
{
    PinnedBuffer<int> nFrozen(1);
    
    PVview view(pv, pv->local());
    const int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    nFrozen.clear(default_stream);

    SAFE_KERNEL_LAUNCH
        (wall_helpers_kernels::collectFrozen<true>,
         nblocks, nthreads, 0, default_stream,
         view, sdfs, minVal, maxVal, nullptr, nFrozen.devPtr());

    nFrozen.downloadFromDevice(default_stream);

    PinnedBuffer<Particle> frozen(nFrozen[0]);
    info("Freezing %d particles", nFrozen[0]);

    pv->local()->resize(nFrozen[0], default_stream);

    nFrozen.clear(default_stream);
    
    SAFE_KERNEL_LAUNCH
        (wall_helpers_kernels::collectFrozen<false>,
         nblocks, nthreads, 0, default_stream,
         view, sdfs, minVal, maxVal, (float4*)frozen.devPtr(), nFrozen.devPtr());

    CUDA_Check( cudaDeviceSynchronize() );
    std::swap(frozen, pv->local()->coosvels);
}

void freezeParticlesInWall(SDF_basedWall *wall, ParticleVector *pv, float minVal, float maxVal)
{
    CUDA_Check( cudaDeviceSynchronize() );

    DeviceBuffer<float> sdfs(pv->local()->size());
    
    wall->sdfPerParticle(pv->local(), &sdfs, nullptr, default_stream);

    extract_particles(pv, sdfs.devPtr(), minVal, maxVal);
}


void freezeParticlesInWalls(std::vector<SDF_basedWall*> walls, ParticleVector *pv, float minVal, float maxVal)
{
    CUDA_Check( cudaDeviceSynchronize() );

    int n = pv->local()->size();
    DeviceBuffer<float> sdfs(n), sdfs_merged(n);

    const int nthreads = 128;
    const int nblocks = getNblocks(n, nthreads);
    const float safety = 1.f;

    SAFE_KERNEL_LAUNCH
        (wall_helpers_kernels::init_sdf,
         nblocks, nthreads, 0, default_stream,
         n, sdfs_merged.devPtr(), minVal - safety);
    
    
    for (auto& wall : walls) {
        wall->sdfPerParticle(pv->local(), &sdfs, nullptr, default_stream);

        SAFE_KERNEL_LAUNCH
            (wall_helpers_kernels::merge_sdfs,
             nblocks, nthreads, 0, default_stream,
             n, sdfs.devPtr(), sdfs_merged.devPtr());
    }

    extract_particles(pv, sdfs_merged.devPtr(), minVal, maxVal);
}


void dumpWalls2XDMF(std::vector<SDF_basedWall*> walls, float3 gridH, DomainInfo domain, std::string filename, MPI_Comm cartComm)
{
    CUDA_Check( cudaDeviceSynchronize() );
    CellListInfo gridInfo(gridH, domain.localSize);
    const int n = gridInfo.totcells;

    DeviceBuffer<float> sdfs(n);
    PinnedBuffer<float> sdfs_merged(n);
        
    const int nthreads = 128;
    const int nblocks = getNblocks(n, nthreads);
    const float initial = -1e5;

    SAFE_KERNEL_LAUNCH
        (wall_helpers_kernels::init_sdf,
         nblocks, nthreads, 0, default_stream,
         n, sdfs_merged.devPtr(), initial);
    
    for (auto& wall : walls)
    {
        wall->sdfOnGrid(gridH, &sdfs, 0);

        SAFE_KERNEL_LAUNCH
            (wall_helpers_kernels::merge_sdfs,
             nblocks, nthreads, 0, default_stream,
             n, sdfs.devPtr(), sdfs_merged.devPtr());
    }

    sdfs_merged.downloadFromDevice(0);
    
    XDMF::UniformGrid grid(gridInfo.ncells, gridInfo.h, cartComm);
    XDMF::Channel sdfCh("sdf", (void*)sdfs_merged.hostPtr(), XDMF::Channel::Type::Scalar);
    XDMF::write(filename, &grid, std::vector<XDMF::Channel>{sdfCh}, cartComm);
}


double volumeInsideWalls(std::vector<SDF_basedWall*> walls, DomainInfo domain, MPI_Comm comm, long nSamplesPerRank)
{
    long n = nSamplesPerRank;
    DeviceBuffer<float3> positions(n);
    DeviceBuffer<float> sdfs(n), sdfs_merged(n);
    PinnedBuffer<int> nInside(1);

    const int nthreads = 128;
    const int nblocks = getNblocks(n, nthreads);
    const float initial = -1e5;

    SAFE_KERNEL_LAUNCH
        (wall_helpers_kernels::initRandomPositions,
         nblocks, nthreads, 0, default_stream,
         n, positions.devPtr(), 424242, domain.localSize);

    SAFE_KERNEL_LAUNCH
        (wall_helpers_kernels::init_sdf,
         nblocks, nthreads, 0, default_stream,
         n, sdfs_merged.devPtr(), initial);
        
    for (auto& wall : walls) {
        wall->sdfPerPosition(&positions, &sdfs, default_stream);

        SAFE_KERNEL_LAUNCH
            (wall_helpers_kernels::merge_sdfs,
             nblocks, nthreads, 0, default_stream,
             n, sdfs.devPtr(), sdfs_merged.devPtr());
    }

    nInside.clear(default_stream);
    
    SAFE_KERNEL_LAUNCH
        (wall_helpers_kernels::countInside,
         nblocks, nthreads, 0, default_stream,
         n, sdfs_merged.devPtr(), nInside.devPtr());

    nInside.downloadFromDevice(default_stream, ContainersSynch::Synch);

    float3 localSize = domain.localSize;
    double subDomainVolume = localSize.x * localSize.y * localSize.z;

    double locVolume = (double) nInside[0] / (double) n * subDomainVolume;
    double totVolume = 0;

    MPI_Check( MPI_Allreduce(&locVolume, &totVolume, 1, MPI_DOUBLE, MPI_SUM, comm) );
    
    return totVolume;
}
