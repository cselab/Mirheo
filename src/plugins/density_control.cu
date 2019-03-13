#include "density_control.h"

#include <core/field/from_function.h>
#include <core/field/utils.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/kernel_launch.h>
#include <core/utils/make_unique.h>

namespace DensityControlPluginKernels
{

enum {INVALID_LEVEL=-1};

__device__ int getLevelId(const FieldDeviceHandler& field, const float3& r,
                           const DensityControlPlugin::LevelBounds& lb)
{
    float l = field(r);
    return (l > lb.lo && l < lb.hi) ?
        (l - lb.lo) / lb.space :
        INVALID_LEVEL;
}

__global__ void countInsideRegions(int nSamples, DomainInfo domain, FieldDeviceHandler field, DensityControlPlugin::LevelBounds lb,
                                   float seed, unsigned long long int *nInsides)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= nSamples) return;
    
    float3 r {Saru::uniform01(seed, i - 2, i + 4242),
              Saru::uniform01(seed, i - 3, i + 4343),
              Saru::uniform01(seed, i - 4, i + 4444)};

    r = domain.localSize * (r - 0.5f);

    int levelId = getLevelId(field, r, lb);
    
    if (levelId != INVALID_LEVEL)
        atomicAdd(&nInsides[levelId], 1);
}

__global__ void computeVolumes(int nLevels, int nSamples, const unsigned long long int *nInsides, double subdomainVolume, double *volumes)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nLevels) return;

    double v = subdomainVolume * (double) nInsides[i] / (double) nSamples;
    volumes[i] = v;
}

__global__ void collectSamples(PVview view, FieldDeviceHandler field, DensityControlPlugin::LevelBounds lb, unsigned long long int *nInsides)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= view.size) return;

    Particle p;
    p.readCoordinate(view.particles, i);

    int levelId = getLevelId(field, p.r, lb);

    if (levelId != INVALID_LEVEL)
        atomicAdd(&nInsides[levelId], 1);
}

__global__ void applyForces(PVview view, FieldDeviceHandler field, DensityControlPlugin::LevelBounds lb, const float *forces)
{
    const float h = 0.25f;
    const float zeroTolerance = 1e-10f;
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= view.size) return;

    Particle p;    
    p.readCoordinate(view.particles, i);

    int levelId = getLevelId(field, p.r, lb);

    if (levelId == INVALID_LEVEL) return;

    float forceMagn = forces[levelId];

    float3 grad = computeGradient(field, p.r, h);

    if (dot(grad, grad) < zeroTolerance) return;

    float3 force = normalize(grad) * forceMagn;

    atomicAdd(view.forces + i, force);
}

} // namespace DensityControlPluginKernels

DensityControlPlugin::DensityControlPlugin(const YmrState *state, std::string name,
                                           std::vector<std::string> pvNames, float targetDensity,
                                           RegionFunc region, float3 resolution,
                                           float levelLo, float levelHi, float levelSpace,
                                           float Kp, float Ki, float Kd,
                                           int tuneEvery, int sampleEvery) :
    SimulationPlugin(state, name),
    pvNames(pvNames),
    targetDensity(targetDensity),
    spaceDecompositionField(std::make_unique<FieldFromFunction>
                            (state, name + "_decomposition", region, resolution)),
    levelBounds({levelLo, levelHi, levelSpace}),
    Kp(Kp), Ki(Ki), Kd(Kd),
    tuneEvery(tuneEvery),
    sampleEvery(sampleEvery),
    nSamples(0)
{}

DensityControlPlugin::~DensityControlPlugin() = default;

void DensityControlPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    for (auto &pvName : pvNames)
        pvs.push_back(simulation->getPVbyNameOrDie(pvName));

    spaceDecompositionField->setup(comm);

    int nLevelSets = (levelBounds.hi - levelBounds.lo) / levelBounds.space;
    levelBounds.space = (levelBounds.hi - levelBounds.lo) / nLevelSets;
    
    nInsides     .resize_anew(nLevelSets);    
    forces       .resize_anew(nLevelSets);

    const float initError = 0;
    controllers.assign(nLevelSets, PidControl<float>(initError, Kp, Ki, Kd));

    volumes   .resize(nLevelSets);
    densities .resize(nLevelSets);
    densities .assign(nLevelSets, 0.0f);
    
    computeVolumes(defaultStream, 1000000);

    nInsides  .clearDevice(defaultStream);    
    forces    .clearDevice(defaultStream);
}


void DensityControlPlugin::beforeForces(cudaStream_t stream)
{
    if (state->currentStep % tuneEvery == 0)
        updatePids(stream);

    if (state->currentStep % sampleEvery == 0)
        sample(stream);

    applyForces(stream);
}

void DensityControlPlugin::computeVolumes(cudaStream_t stream, int MCnSamples)
{
    const int nthreads = 128;
    float seed = 0.42424242f + rank * 17;
    auto domain = state->domain;    
    int nLevelSets = nInsides.size();

    PinnedBuffer<double> localVolumes(nLevelSets);
    
    nInsides    .clearDevice(stream);
    localVolumes.clearDevice(stream);
    
    SAFE_KERNEL_LAUNCH(
        DensityControlPluginKernels::countInsideRegions,
        getNblocks(MCnSamples, nthreads), nthreads, 0, stream,
        MCnSamples, domain, spaceDecompositionField->handler(),
        levelBounds, seed, nInsides.devPtr());

    float3 L = domain.localSize;
    double subdomainVolume = L.x * L.y * L.z;

    SAFE_KERNEL_LAUNCH(
        DensityControlPluginKernels::computeVolumes,
        getNblocks(localVolumes.size(), nthreads), nthreads, 0, stream,
        localVolumes.size(), MCnSamples, nInsides.devPtr(),
        subdomainVolume, localVolumes.devPtr());

    volumes.resize(nLevelSets);
    volumes.assign(nLevelSets, 0.0);
    
    localVolumes.downloadFromDevice(stream);
    
    MPI_Check( MPI_Allreduce(localVolumes.hostPtr(), volumes.data(), volumes.size(), MPI_DOUBLE, MPI_SUM, comm) );
}

void DensityControlPlugin::sample(cudaStream_t stream)
{
    const int nthreads = 128;

    for (auto pv : pvs)
    {
        PVview view(pv, pv->local());

        SAFE_KERNEL_LAUNCH(
            DensityControlPluginKernels::collectSamples,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, spaceDecompositionField->handler(),
            levelBounds, nInsides.devPtr());        
    }

    ++nSamples;
}

void DensityControlPlugin::updatePids(cudaStream_t stream)
{
    nInsides.downloadFromDevice(stream);    
    
    MPI_Check( MPI_Allreduce(MPI_IN_PLACE, nInsides.hostPtr(), nInsides.size(),
                             MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm) );

    for (int i = 0; i < volumes.size(); ++i)
    {
        double denom = volumes[i] * nSamples;

        densities[i] = (denom > 1e-6) ? 
            nInsides[i] / denom :
            0.0;
    }       

    for (int i = 0; i < densities.size(); ++i)
    {
        float rhom = i > 0 ? densities[i-1] : targetDensity;
        float rho  = densities[i];
        float error = (rho - rhom) / levelBounds.space;
        
        forces[i] = controllers[i].update(error);
    }

    forces.uploadToDevice(stream);
    
    nInsides.clearDevice(stream);
    nSamples = 0;
}

void DensityControlPlugin::applyForces(cudaStream_t stream)
{
    const int nthreads = 128;
    
    for (auto pv : pvs)
    {
        PVview view(pv, pv->local());

        SAFE_KERNEL_LAUNCH(
            DensityControlPluginKernels::applyForces,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, spaceDecompositionField->handler(),
            levelBounds, forces.devPtr());        
    }    
}

