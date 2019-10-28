#include "wall_helpers.h"

#include <core/celllist.h>
#include <core/logger.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/utils/cuda_common.h>
#include <core/utils/folders.h>
#include <core/utils/kernel_launch.h>
#include <core/walls/simple_stationary_wall.h>
#include <core/xdmf/type_map.h>
#include <core/xdmf/xdmf.h>

#include <curand_kernel.h>

namespace WallHelpersKernels
{
__global__ void init_sdf(int n, real *sdfs, real val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sdfs[i] = val;
}

__global__ void merge_sdfs(int n, const real *sdfs, real *sdfs_merged)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sdfs_merged[i] = max(sdfs[i], sdfs_merged[i]);
}

template<bool QUERY>
__global__ void collectFrozen(PVview view, const real *sdfs, real minVal, real maxVal,
                              real4 *frozenPos, real4 *frozenVel, int *nFrozen)
{
    const int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view.size) return;

    Particle p(view.readParticle(pid));
    p.u = make_real3(0);

    const real val = sdfs[pid];
        
    if (val > minVal && val < maxVal)
    {
        const int ind = atomicAggInc(nFrozen);

        if (!QUERY)
            p.write2Real4(frozenPos, frozenVel, ind);
    }
}

__global__ void initRandomPositions(int n, real3 *positions, long seed, real3 localSize)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;
        
    curandState_t state;
    real3 r;
        
    curand_init(seed, i, 0, &state);
        
    r.x = localSize.x * (curand_uniform(&state) - 0.5_r);
    r.y = localSize.y * (curand_uniform(&state) - 0.5_r);
    r.z = localSize.z * (curand_uniform(&state) - 0.5_r);

    positions[i] = r;
}

__global__ void countInside(int n, const real *sdf, int *nInside, real threshold = 0._r)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    int myval = 0;

    if (i < n)
        myval = sdf[i] < threshold;

    myval = warpReduce(myval, [] (int a, int b) {return a + b;});

    if (laneId() == 0)
        atomicAdd(nInside, myval);
}
} // namespace WallHelpersKernels

static void extract_particles(ParticleVector *pv, const real *sdfs, real minVal, real maxVal)
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

    PinnedBuffer<real4> frozenPos(nFrozen[0]), frozenVel(nFrozen[0]);
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

void WallHelpers::freezeParticlesInWall(SDF_basedWall *wall, ParticleVector *pv, real minVal, real maxVal)
{
    CUDA_Check( cudaDeviceSynchronize() );

    DeviceBuffer<real> sdfs(pv->local()->size());
    
    wall->sdfPerParticle(pv->local(), &sdfs, nullptr, 0, defaultStream);

    extract_particles(pv, sdfs.devPtr(), minVal, maxVal);
}


void WallHelpers::freezeParticlesInWalls(std::vector<SDF_basedWall*> walls, ParticleVector *pv, real minVal, real maxVal)
{
    CUDA_Check( cudaDeviceSynchronize() );

    int n = pv->local()->size();
    DeviceBuffer<real> sdfs(n), sdfs_merged(n);

    const int nthreads = 128;
    const int nblocks = getNblocks(n, nthreads);
    const real safety = 1._r;

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


void WallHelpers::dumpWalls2XDMF(std::vector<SDF_basedWall*> walls, real3 gridH, DomainInfo domain, std::string filename, MPI_Comm cartComm)
{
    CUDA_Check( cudaDeviceSynchronize() );
    CellListInfo gridInfo(gridH, domain.localSize);
    const int n = gridInfo.totcells;

    DeviceBuffer<real> sdfs(n);
    PinnedBuffer<real> sdfs_merged(n);
        
    const int nthreads = 128;
    const int nblocks = getNblocks(n, nthreads);
    const real initial = -1e5;

    SAFE_KERNEL_LAUNCH(
        WallHelpersKernels::init_sdf,
        nblocks, nthreads, 0, defaultStream,
        n, sdfs_merged.devPtr(), initial);
    
    for (auto& wall : walls)
    {
        wall->sdfOnGrid(gridH, &sdfs, defaultStream);

        SAFE_KERNEL_LAUNCH(
            WallHelpersKernels::merge_sdfs,
            nblocks, nthreads, 0, defaultStream,
            n, sdfs.devPtr(), sdfs_merged.devPtr());
    }

    sdfs_merged.downloadFromDevice(defaultStream);

    auto path = parentPath(filename);
    if (path != filename)
        createFoldersCollective(cartComm, path);
    
    XDMF::UniformGrid grid(gridInfo.ncells, gridInfo.h, cartComm);
    XDMF::Channel sdfCh("sdf", sdfs_merged.hostPtr(),
                        XDMF::Channel::DataForm::Scalar,
                        XDMF::getNumberType<real>(),
                        DataTypeWrapper<real>(),
                        XDMF::Channel::NeedShift::False);
    XDMF::write(filename, &grid, {sdfCh}, cartComm);
}


double WallHelpers::volumeInsideWalls(std::vector<SDF_basedWall*> walls, DomainInfo domain, MPI_Comm comm, long nSamplesPerRank)
{
    long n = nSamplesPerRank;
    DeviceBuffer<real3> positions(n);
    DeviceBuffer<real> sdfs(n), sdfs_merged(n);
    PinnedBuffer<int> nInside(1);

    const int nthreads = 128;
    const int nblocks = getNblocks(n, nthreads);
    const real initial = -1e5;

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

    real3  localSize = domain.localSize;
    double subDomainVolume = localSize.x * localSize.y * localSize.z;

    double locVolume = (double) nInside[0] / (double) n * subDomainVolume;
    double totVolume = 0;

    MPI_Check( MPI_Allreduce(&locVolume, &totVolume, 1, MPI_DOUBLE, MPI_SUM, comm) );
    
    return totVolume;
}
