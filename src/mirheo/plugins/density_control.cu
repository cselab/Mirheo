// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "density_control.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/field/from_function.h>
#include <mirheo/core/field/utils.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/kernel_launch.h>

#include <fstream>
#include <memory>

namespace mirheo
{

namespace density_control_plugin_kernels
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

} // namespace density_control_plugin_kernels

DensityControlPlugin::DensityControlPlugin(const MirState *state, std::string name,
                                           std::vector<std::string> pvNames, real targetDensity,
                                           RegionFunc region, real3 resolution,
                                           real levelLo, real levelHi, real levelSpace,
                                           real Kp, real Ki, real Kd,
                                           int tuneEvery, int dumpEvery, int sampleEvery) :
    SimulationPlugin(state, name),
    pvNames_(pvNames),
    targetDensity_(targetDensity),
    spaceDecompositionField_(std::make_unique<FieldFromFunction>
                            (state, name + "_decomposition", region, resolution)),
    levelBounds_({levelLo, levelHi, levelSpace}),
    Kp_(Kp), Ki_(Ki), Kd_(Kd),
    tuneEvery_(tuneEvery),
    dumpEvery_(dumpEvery),
    sampleEvery_(sampleEvery),
    nSamples_(0)
{}

DensityControlPlugin::~DensityControlPlugin() = default;

void DensityControlPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    if (nSamples_ > 0) {
        // Ideally, running `run` multiple times should produce exactly the
        // same result as running a single large `run`. However, it is not
        // clear what should happens if new particle vectors are added or
        // removed between two runs.
        die("DensityControlPlugin does not support multiple runs.");
    }

    pvs_.clear();
    for (auto &pvName : pvNames_)
        pvs_.push_back(simulation->getPVbyNameOrDie(pvName));

    spaceDecompositionField_->setup(comm);

    const int nLevelSets = (levelBounds_.hi - levelBounds_.lo) / levelBounds_.space;
    levelBounds_.space = (levelBounds_.hi - levelBounds_.lo) / nLevelSets;

    nInsides_ .resize_anew(nLevelSets);
    forces_   .resize_anew(nLevelSets);

    const real initError = 0;
    controllers_.assign(nLevelSets, PidControl<real>(initError, Kp_, Ki_, Kd_));

    volumes_  .resize(nLevelSets);
    densities_.resize(nLevelSets);
    densities_.assign(nLevelSets, 0.0_r);

    computeVolumes(defaultStream, 1000000);

    nInsides_ .clearDevice(defaultStream);
    forces_   .clearDevice(defaultStream);
    nSamples_ = 0;
}


void DensityControlPlugin::beforeForces(cudaStream_t stream)
{
    if (isTimeEvery(getState(), tuneEvery_))
        updatePids(stream);

    if (isTimeEvery(getState(), sampleEvery_))
        sample(stream);

    applyForces(stream);
}

void DensityControlPlugin::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (!isTimeEvery(getState(), dumpEvery_)) return;

    _waitPrevSend();
    SimpleSerializer::serialize(sendBuffer_, getState()->currentTime, getState()->currentStep, densities_, forces_);
    _send(sendBuffer_);
}


void DensityControlPlugin::computeVolumes(cudaStream_t stream, int MCnSamples)
{
    const int nthreads = 128;
    const real seed = 0.42424242_r + rank_ * 17;
    const auto domain = getState()->domain;
    const int nLevelSets = nInsides_.size();

    PinnedBuffer<double> localVolumes(nLevelSets);

    nInsides_    .clearDevice(stream);
    localVolumes.clearDevice(stream);

    SAFE_KERNEL_LAUNCH(
        density_control_plugin_kernels::countInsideRegions,
        getNblocks(MCnSamples, nthreads), nthreads, 0, stream,
        MCnSamples, domain, spaceDecompositionField_->handler(),
        levelBounds_, seed, nInsides_.devPtr());

    const real3 L = domain.localSize;
    const double subdomainVolume = L.x * L.y * L.z;

    SAFE_KERNEL_LAUNCH(
        density_control_plugin_kernels::computeVolumes,
        getNblocks(localVolumes.size(), nthreads), nthreads, 0, stream,
        localVolumes.size(), MCnSamples, nInsides_.devPtr(),
        subdomainVolume, localVolumes.devPtr());

    volumes_.resize(nLevelSets);
    volumes_.assign(nLevelSets, 0.0);

    localVolumes.downloadFromDevice(stream);

    MPI_Check( MPI_Allreduce(localVolumes.hostPtr(), volumes_.data(), volumes_.size(), MPI_DOUBLE, MPI_SUM, comm_) );
    // std::copy(localVolumes.begin(), localVolumes.end(), volumes.begin());
}

void DensityControlPlugin::sample(cudaStream_t stream)
{
    const int nthreads = 128;

    for (auto pv : pvs_)
    {
        PVview view(pv, pv->local());

        SAFE_KERNEL_LAUNCH(
            density_control_plugin_kernels::collectSamples,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, spaceDecompositionField_->handler(),
            levelBounds_, nInsides_.devPtr());
    }

    ++nSamples_;
}

void DensityControlPlugin::updatePids(cudaStream_t stream)
{
    nInsides_.downloadFromDevice(stream);

    MPI_Check( MPI_Allreduce(MPI_IN_PLACE, nInsides_.hostPtr(), nInsides_.size(),
                             MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm_) );

    for (size_t i = 0; i < volumes_.size(); ++i)
    {
        const double denom = volumes_[i] * nSamples_;

        densities_[i] = (denom > 1e-6) ?
            nInsides_[i] / denom :
            0.0;
    }

    for (size_t i = 0; i < densities_.size(); ++i)
    {
        const real error = densities_[i] - targetDensity_;
        forces_[i] = controllers_[i].update(error);
    }

    forces_.uploadToDevice(stream);

    nInsides_.clearDevice(stream);
    nSamples_ = 0;
}

void DensityControlPlugin::applyForces(cudaStream_t stream)
{
    const int nthreads = 128;

    for (auto pv : pvs_)
    {
        PVview view(pv, pv->local());

        SAFE_KERNEL_LAUNCH(
            density_control_plugin_kernels::applyForces,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, spaceDecompositionField_->handler(),
            levelBounds_, forces_.devPtr());
    }
}

void DensityControlPlugin::checkpoint(MPI_Comm comm, const std::string& path, int checkpointId)
{
    const auto filename = createCheckpointNameWithId(path, "plugin." + getName(), "txt", checkpointId);

    {
        std::ofstream fout(filename);
        for (const auto& pid : controllers_)
            fout << pid << std::endl;
    }

    createCheckpointSymlink(comm, path, "plugin." + getName(), "txt", checkpointId);
}

void DensityControlPlugin::restart(__UNUSED MPI_Comm comm, const std::string& path)
{
    const auto filename = createCheckpointName(path, "plugin." + getName(), "txt");

    std::ifstream fin(filename);

    for (auto& pid : controllers_)
        fin >> pid;
}




PostprocessDensityControl::PostprocessDensityControl(std::string name, std::string filename) :
    PostprocessPlugin(name)
{
    auto status = fdump_.open(filename, "w");
    if (status != FileWrapper::Status::Success)
        die("Could not open file '%s'", filename.c_str());
}

void PostprocessDensityControl::deserialize()
{
    MirState::StepType currentTimeStep;
    MirState::TimeType currentTime;
    std::vector<real> densities, forces;

    SimpleSerializer::deserialize(data_, currentTime, currentTimeStep, densities, forces);

    if (rank_ == 0)
    {
        fprintf(fdump_.get(), "%g %lld ", currentTime, currentTimeStep);
        for (auto d : densities) fprintf(fdump_.get(), "%g ", d);
        for (auto f : forces)    fprintf(fdump_.get(), "%g ", f);
        fprintf(fdump_.get(), "\n");

        fflush(fdump_.get());
    }
}

} // namespace mirheo
