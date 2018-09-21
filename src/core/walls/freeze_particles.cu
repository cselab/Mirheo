#include "freeze_particles.h"

#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cuda_common.h>
#include <core/celllist.h>
#include <core/utils/kernel_launch.h>
#include <core/xdmf/xdmf.h>

#include <core/walls/simple_stationary_wall.h>

static const cudaStream_t default_stream = 0;

namespace freeze_particles_kernels
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
}

static void extract_particles(ParticleVector *pv, const float *sdfs, float minVal, float maxVal)
{
    PinnedBuffer<int> nFrozen(1);
    
    PVview view(pv, pv->local());
    const int nthreads = 128;
    const int nblocks = getNblocks(view.size, nthreads);

    nFrozen.clear(default_stream);

    SAFE_KERNEL_LAUNCH
        (freeze_particles_kernels::collectFrozen<true>,
         nblocks, nthreads, 0, default_stream,
         view, sdfs, minVal, maxVal, nullptr, nFrozen.devPtr());

    nFrozen.downloadFromDevice(default_stream);

    PinnedBuffer<Particle> frozen(nFrozen[0]);
    info("Freezing %d particles", nFrozen[0]);

    pv->local()->resize(nFrozen[0], default_stream);

    nFrozen.clear(default_stream);
    
    SAFE_KERNEL_LAUNCH
        (freeze_particles_kernels::collectFrozen<false>,
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
        (freeze_particles_kernels::init_sdf,
         nblocks, nthreads, 0, default_stream,
         n, sdfs_merged.devPtr(), minVal - safety);
    
    
    for (auto& wall : walls) {
        wall->sdfPerParticle(pv->local(), &sdfs, nullptr, default_stream);

        SAFE_KERNEL_LAUNCH
            (freeze_particles_kernels::merge_sdfs,
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
        (freeze_particles_kernels::init_sdf,
         nblocks, nthreads, 0, default_stream,
         n, sdfs_merged.devPtr(), initial);
    
    for (auto& wall : walls)
    {
        wall->sdfOnGrid(gridH, &sdfs, 0);

        SAFE_KERNEL_LAUNCH
            (freeze_particles_kernels::merge_sdfs,
             nblocks, nthreads, 0, default_stream,
             n, sdfs.devPtr(), sdfs_merged.devPtr());
    }

    sdfs_merged.downloadFromDevice(0);
    
    XDMF::UniformGrid grid(gridInfo.ncells, gridInfo.h, cartComm);
    XDMF::Channel sdfCh("sdf", (void*)sdfs_merged.hostPtr(), XDMF::Channel::Type::Scalar, sizeof(float), "float1");
    XDMF::write(filename, &grid, std::vector<XDMF::Channel>{sdfCh}, cartComm);
}

