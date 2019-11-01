#include "density_control.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/field/from_function.h>
#include <core/field/utils.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/kernel_launch.h>

#include <fstream>
#include <memory>

namespace DensityControlPluginKernels
{

enum {INVALID_LEVEL=-1};

__device__ int getLevelId(const FieldDeviceHandler& field, const real3& r,
                           const DensityControlPlugin::LevelBounds& lb)
{
    real l = field(r);
    return (l > lb.lo && l < lb.hi) ?
        (l - lb.lo) / lb.space :
        INVALID_LEVEL;
}

__global__ void countInsideRegions(int nSamples, DomainInfo domain, FieldDeviceHandler field, DensityControlPlugin::LevelBounds lb,
                                   real seed, unsigned long long int *nInsides)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= nSamples) return;
    
    real3 r {Saru::uniform01(seed, i - 2, i + 4242),
             Saru::uniform01(seed, i - 3, i + 4343),
             Saru::uniform01(seed, i - 4, i + 4444)};

    r = domain.localSize * (r - 0._r);

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

    auto r = Real3_int(view.readPosition(i)).v;

    int levelId = getLevelId(field, r, lb);

    if (levelId != INVALID_LEVEL)
        atomicAdd(&nInsides[levelId], 1);
}

__global__ void applyForces(PVview view, FieldDeviceHandler field, DensityControlPlugin::LevelBounds lb, const real *forces)
{
    const real h = 0.25_r;
    const real zeroTolerance = 1e-10_r;
    
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= view.size) return;

    auto r = Real3_int(view.readPosition(i)).v;

    int levelId = getLevelId(field, r, lb);

    if (levelId == INVALID_LEVEL) return;

    real forceMagn = forces[levelId];

    real3 grad = computeGradient(field, r, h);

    if (dot(grad, grad) < zeroTolerance) return;

    real3 force = normalize(grad) * forceMagn;

    atomicAdd(view.forces + i, force);
}

} // namespace DensityControlPluginKernels

DensityControlPlugin::DensityControlPlugin(const MirState *state, std::string name,
                                           std::vector<std::string> pvNames, real targetDensity,
                                           RegionFunc region, real3 resolution,
                                           real levelLo, real levelHi, real levelSpace,
                                           real Kp, real Ki, real Kd,
                                           int tuneEvery, int dumpEvery, int sampleEvery) :
    SimulationPlugin(state, name),
    pvNames(pvNames),
    targetDensity(targetDensity),
    spaceDecompositionField(std::make_unique<FieldFromFunction>
                            (state, name + "_decomposition", region, resolution)),
    levelBounds({levelLo, levelHi, levelSpace}),
    Kp(Kp), Ki(Ki), Kd(Kd),
    tuneEvery(tuneEvery),
    dumpEvery(dumpEvery),
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

    const real initError = 0;
    controllers.assign(nLevelSets, PidControl<real>(initError, Kp, Ki, Kd));

    volumes   .resize(nLevelSets);
    densities .resize(nLevelSets);
    densities .assign(nLevelSets, 0.0_r);
    
    computeVolumes(defaultStream, 1000000);

    nInsides  .clearDevice(defaultStream);    
    forces    .clearDevice(defaultStream);
    nSamples = 0;
}


void DensityControlPlugin::beforeForces(cudaStream_t stream)
{
    if (isTimeEvery(state, tuneEvery))
        updatePids(stream);

    if (isTimeEvery(state, sampleEvery))
        sample(stream);

    applyForces(stream);
}

void DensityControlPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!isTimeEvery(state, dumpEvery)) return;

    waitPrevSend();
    SimpleSerializer::serialize(sendBuffer, state->currentTime, state->currentStep, densities, forces);
    send(sendBuffer);
}


void DensityControlPlugin::computeVolumes(cudaStream_t stream, int MCnSamples)
{
    const int nthreads = 128;
    real seed = 0.42424242_r + rank * 17;
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

    real3 L = domain.localSize;
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
    // std::copy(localVolumes.begin(), localVolumes.end(), volumes.begin());
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

    for (size_t i = 0; i < volumes.size(); ++i)
    {
        const double denom = volumes[i] * nSamples;

        densities[i] = (denom > 1e-6) ? 
            nInsides[i] / denom :
            0.0;
    }       

    for (size_t i = 0; i < densities.size(); ++i)
    {
        const real error = densities[i] - targetDensity;        
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

void DensityControlPlugin::checkpoint(MPI_Comm comm, const std::string& path, int checkpointId)
{
    const auto filename = createCheckpointNameWithId(path, "plugin." + name, "txt", checkpointId);

    {
        std::ofstream fout(filename);
        for (const auto& pid : controllers)
            fout << pid << std::endl;
    }
    
    createCheckpointSymlink(comm, path, "plugin." + name, "txt", checkpointId);
}

void DensityControlPlugin::restart(__UNUSED MPI_Comm comm, const std::string& path)
{
    const auto filename = createCheckpointName(path, "plugin." + name, "txt");

    std::ifstream fin(filename);

    for (auto& pid : controllers)
        fin >> pid;
}




PostprocessDensityControl::PostprocessDensityControl(std::string name, std::string filename) :
    PostprocessPlugin(name)
{
    auto status = fdump.open(filename, "w");
    if (status != FileWrapper::Status::Success)
        die("Could not open file '%s'", filename.c_str());
}

void PostprocessDensityControl::deserialize()
{
    MirState::StepType currentTimeStep;
    MirState::TimeType currentTime;
    std::vector<real> densities, forces;

    SimpleSerializer::deserialize(data, currentTime, currentTimeStep, densities, forces);

    if (rank == 0)
    {
        fprintf(fdump.get(), "%g %lld ", currentTime, currentTimeStep);
        for (auto d : densities) fprintf(fdump.get(), "%g ", d);
        for (auto f : forces)    fprintf(fdump.get(), "%g ", f);
        fprintf(fdump.get(), "\n");
        
        fflush(fdump.get());
    }
}
