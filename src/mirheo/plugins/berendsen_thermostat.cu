#include "berendsen_thermostat.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/config.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/mpi_types.h>

namespace mirheo
{

namespace BerendsenThermostatKernels
{

/// Compute the sum of (m * vx, m * vy, m * vz, m * v^2 / 2) for a given PV.
__global__ void reduceVelocityAndEnergy(PVview view, real4 *stats)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    real4 warpStats{0, 0, 0, 0};
    real4 threadStats{0, 0, 0, 0};

    if (tid < view.size) {
        threadStats   = view.readVelocity(tid);
        real3 vel     = make_real3(threadStats);
        threadStats.w = dot(vel, vel);
    }

    warpStats = warpReduce(threadStats, [](real a, real b) { return a + b; });

    if (laneId() == 0)
        atomicAdd(stats, warpStats);
}

// Update velocities v[i] := avgv + lambda * (v[i] - avgv).
__global__ void updateVelocities(PVview view, real3 avgVel, real lambda)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= view.size)
        return;

    real4 vel4 = view.readVelocity(tid);
    real3 vel3 = make_real3(vel4);
    vel3 = avgVel + lambda * (vel3 - avgVel);
    view.writeVelocity(tid, make_real4(vel3, vel4.w));
}

} // namespace BerendsenThermostatKernels


BerendsenThermostatPlugin::BerendsenThermostatPlugin(
        const MirState *state, std::string name, std::vector<std::string> pvNames,
        real kBT, real tau, bool increaseIfLower) :
    SimulationPlugin(state, name),
    pvNames_(std::move(pvNames)),
    kBT_(kBT),
    tau_(tau),
    increaseIfLower_(increaseIfLower),
    stats_(pvNames_.size())
{}

BerendsenThermostatPlugin::BerendsenThermostatPlugin(
        const MirState *state, Loader& loader, const ConfigObject& config) :
    BerendsenThermostatPlugin{
        state, config["name"], loader.load<std::vector<std::string>>(config["pvNames"]),
        config["kBT"], config["tau"], config["increaseIfLower"]}
{}

void BerendsenThermostatPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pvs_.reserve(pvNames_.size());
    for (const std::string& pvName : pvNames_)
        pvs_.push_back(simulation->getPVbyNameOrDie(pvName));
}

void BerendsenThermostatPlugin::afterIntegration(cudaStream_t stream)
{
    const int nthreads = 128;

    // Gather statistics for each individual PV.
    stats_.clearDevice(stream);
    for (size_t i = 0; i < pvs_.size(); ++i) {
        PVview view(pvs_[i], pvs_[i]->local());
        SAFE_KERNEL_LAUNCH(BerendsenThermostatKernels::reduceVelocityAndEnergy,
                           getNblocks(view.size, nthreads), nthreads, 0, stream,
                           view, stats_.devPtr() + i);
    }
    stats_.downloadFromDevice(stream);

    // Reduce local PV sums of (m * v, m * v^2, m, 1).
    real localTotal[6] = {0, 0, 0, 0, 0, 0};
    for (size_t i = 0; i < pvs_.size(); ++i) {
        real mass = pvs_[i]->getMassPerParticle();
        localTotal[0] += mass * stats_[i].x;
        localTotal[1] += mass * stats_[i].y;
        localTotal[2] += mass * stats_[i].z;
        localTotal[3] += mass * stats_[i].w;
        localTotal[4] += mass * pvs_[i]->local()->size();
        localTotal[5] += pvs_[i]->local()->size();
    }

    // Reduce global results.
    real total[6] = {0, 0, 0, 0, 0, 0};
    MPI_Check( MPI_Allreduce(&localTotal[0], &total[0], 6,
                             getMPIFloatType<real>(), MPI_SUM, comm_) );

    // Compute average velocity lambda.
    using std::sqrt;
    real totalParticles = total[5];
    real totalMass = total[4];
    real3 avgVel = make_real3(total[0], total[1], total[2]) / totalMass;
    real kineticEnergy = 0.5_r * (total[3] - totalMass * dot(avgVel, avgVel));
    // E_kinetic == 1/2 * k_B * T * <number of degrees of freedom>
    real currentkBT = kineticEnergy * 2 / (3 * totalParticles);
    real lambda = increaseIfLower_ || currentkBT > kBT_
            ? sqrt(1 + getState()->dt / tau_ * (kBT_ / currentkBT - 1))
            : 1.0_r;

    // Update local particles.
    for (ParticleVector *pv : pvs_) {
        PVview view(pv, pv->local());
        SAFE_KERNEL_LAUNCH(BerendsenThermostatKernels::updateVelocities,
                           getNblocks(view.size, nthreads), nthreads, 0, stream,
                           view, avgVel, lambda);
    }
}

void BerendsenThermostatPlugin::saveSnapshotAndRegister(Saver& saver)
{
    saver.registerObject<BerendsenThermostatPlugin>(
            this, _saveSnapshot(saver, "BerendsenThermostatPlugin"));
}

ConfigObject BerendsenThermostatPlugin::_saveSnapshot(Saver& saver, const std::string& typeName)
{
    ConfigObject config = SimulationPlugin::_saveSnapshot(saver, typeName);
    config.emplace("pvNames",         saver(pvNames_));
    config.emplace("kBT",             saver(kBT_));
    config.emplace("tau",             saver(tau_));
    config.emplace("increaseIfLower", saver(increaseIfLower_));
    return config;
}

} // namespace mirheo
