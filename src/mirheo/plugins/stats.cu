// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "stats.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <mirheo/core/datatypes.h>
#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/mpi_types.h>
#include <mirheo/core/utils/path.h>

namespace mirheo
{

namespace stats_plugin_kernels
{
using stats_plugin::ReductionType;

__global__ void totalMomentumEnergy(PVview view, ReductionType *momentum, ReductionType *energy, real *maxvel)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    real3 vel, myMomentum;
    real myEnergy = 0._r, myMaxIvelI;
    vel = myMomentum = make_real3(0._r);

    if (tid < view.size)
    {
        vel        = make_real3(view.readVelocity(tid));
        myMomentum = vel * view.mass;
        myEnergy   = dot(vel, vel) * view.mass * 0.5_r;
    }

    myMomentum = warpReduce(myMomentum, [](real a, real b) { return a+b; });
    myEnergy   = warpReduce(myEnergy,   [](real a, real b) { return a+b; });

    myMaxIvelI = warpReduce(length(vel), [](real a, real b) { return math::max(a, b); });

    if (laneId() == 0)
    {
        atomicAdd(momentum+0, (ReductionType)myMomentum.x);
        atomicAdd(momentum+1, (ReductionType)myMomentum.y);
        atomicAdd(momentum+2, (ReductionType)myMomentum.z);
        atomicAdd(energy,     (ReductionType)myEnergy);

        atomicMax((int*)maxvel, __float_as_int(myMaxIvelI));
    }
}
} // namespace stats_plugin_kernels

SimulationStats::SimulationStats(const MirState *state, std::string name, int every, std::vector<std::string> pvNames) :
    SimulationPlugin(state, name),
    every_(every),
    pvNames_(std::move(pvNames))
{}

SimulationStats::~SimulationStats() = default;

void SimulationStats::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    if (pvNames_.empty())
    {
        pvs_ = simulation->getParticleVectors();
    }
    else
    {
        for (const auto& pvName : pvNames_)
            pvs_.push_back(simulation->getPVbyNameOrDie(pvName));
    }

    timer_.start();
}

void SimulationStats::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(getState(), every_))
        return;

    momentum_.clear(stream);
    energy_  .clear(stream);
    maxvel_  .clear(stream);

    nparticles_ = 0;
    for (auto& pv : pvs_)
    {
        PVview view(pv, pv->local());
        constexpr int nthreads = 128;

        SAFE_KERNEL_LAUNCH(
                stats_plugin_kernels::totalMomentumEnergy,
                getNblocks(view.size, nthreads), nthreads, 0, stream,
                view, momentum_.devPtr(), energy_.devPtr(), maxvel_.devPtr() );

        nparticles_ += view.size;
    }

    momentum_.downloadFromDevice(stream, ContainersSynch::Asynch);
    energy_  .downloadFromDevice(stream, ContainersSynch::Asynch);
    maxvel_  .downloadFromDevice(stream);

    needToDump_ = true;
}

void SimulationStats::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (needToDump_)
    {
        const real tm = timer_.elapsedAndReset() / (getState()->currentStep < every_ ? 1.0_r : every_);
        _waitPrevSend();
        SimpleSerializer::serialize(sendBuffer_, tm, getState()->currentTime,
                                    getState()->currentStep,
                                    nparticles_, momentum_, energy_, maxvel_);
        _send(sendBuffer_);
        needToDump_ = false;
    }
}

PostprocessStats::PostprocessStats(std::string name, std::string filename) :
    PostprocessPlugin(name),
    filename_(std::move(filename))
{
    if (filename_ != "")
    {
        filename_ = setExtensionOrDie(filename_, "csv");

        const auto status = fdump_.open(filename_, "w");
        if (status != FileWrapper::Status::Success)
            die("Could not open file '%s'", filename_.c_str());

        fprintf(fdump_.get(), "time,kBT,vx,vy,vz,maxv,num_particles,simulation_time_per_step\n");
        fflush(fdump_.get());
    }
}

void PostprocessStats::deserialize()
{
    MirState::TimeType currentTime;
    MirState::StepType currentTimeStep;
    real realTime;
    stats_plugin::CountType nparticles, maxNparticles, minNparticles;

    std::vector<stats_plugin::ReductionType> momentum, energy;
    std::vector<real> maxvel;

    SimpleSerializer::deserialize(data_, realTime, currentTime, currentTimeStep,
                                  nparticles, momentum, energy, maxvel);

    MPI_Check( MPI_Reduce(&nparticles, &minNparticles, 1, getMPIIntType<stats_plugin::CountType>(), MPI_MIN, 0, comm_) );
    MPI_Check( MPI_Reduce(&nparticles, &maxNparticles, 1, getMPIIntType<stats_plugin::CountType>(), MPI_MAX, 0, comm_) );

    MPI_Check( MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : &nparticles,     &nparticles,     1, getMPIIntType<stats_plugin::CountType>(),       MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : energy.data(),   energy.data(),   1, getMPIFloatType<stats_plugin::ReductionType>(), MPI_SUM, 0, comm_) );
    MPI_Check( MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : momentum.data(), momentum.data(), 3, getMPIFloatType<stats_plugin::ReductionType>(), MPI_SUM, 0, comm_) );

    MPI_Check( MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : maxvel.data(),   maxvel.data(),   1, getMPIFloatType<real>(), MPI_MAX, 0, comm_) );

    MPI_Check( MPI_Reduce(rank_ == 0 ? MPI_IN_PLACE : &realTime,       &realTime,       1, getMPIFloatType<real>(), MPI_MAX, 0, comm_) );

    if (rank_ == 0)
    {
        const double invNparticles = nparticles > 0 ? 1.0 / nparticles : 0.0;
        momentum[0] *= invNparticles;
        momentum[1] *= invNparticles;
        momentum[2] *= invNparticles;
        const stats_plugin::ReductionType kBT = energy[0] * invNparticles * (2.0/3.0);

        printf("Stats at timestep %lld (simulation time %f):\n", currentTimeStep, currentTime);
        printf("\tOne timestep takes %.2f ms", realTime);
        printf("\tNumber of particles (total, min/proc, max/proc): %llu,  %llu,  %llu\n", nparticles, minNparticles, maxNparticles);
        printf("\tAverage momentum: [%e %e %e]\n", momentum[0], momentum[1], momentum[2]);
        printf("\tMax velocity magnitude: %f\n", maxvel[0]);
        printf("\tTemperature: %.4f\n\n", kBT);


        if (fdump_.get())
        {
            fprintf(fdump_.get(), "%g,%g,%g,%g,%g,%g,%llu,%g\n", currentTime,
                    kBT, momentum[0], momentum[1], momentum[2],
                    maxvel[0], nparticles, realTime);
            fflush(fdump_.get());
        }
    }
}

void PostprocessStats::checkpoint(MPI_Comm comm, const std::string& path, int checkpointId)
{
    if (filename_ == "")
        return;

    int rank {0};
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    const auto checkpointFilename = createCheckpointNameWithId(path, "plugin.post." + getName(), "csv", checkpointId);

    // copy current file
    if (rank == 0)
        copyFile(filename_, checkpointFilename);

    MPI_Check( MPI_Barrier(comm) );

    createCheckpointSymlink(comm, path, "plugin.post." + getName(), "csv", checkpointId);
}

void PostprocessStats::restart(MPI_Comm comm, const std::string& path)
{
    if (filename_ == "")
        return;

    int rank {0};
    MPI_Check( MPI_Comm_rank(comm, &rank) );

    if (fdump_.get())
        fdump_.close();

    const auto checkpointFilename = createCheckpointName(path, "plugin.post." + getName(), "csv");

    if (rank == 0)
        copyFile(checkpointFilename, filename_);

    MPI_Check( MPI_Barrier(comm) );

    const auto status = fdump_.open(filename_, "a");
    if (status != FileWrapper::Status::Success)
        die("Could not open file '%s'", filename_.c_str());
}

} // namespace mirheo
