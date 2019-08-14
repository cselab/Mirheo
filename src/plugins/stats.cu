#include "stats.h"
#include "utils/simple_serializer.h"
#include "utils/time_stamp.h"

#include <core/datatypes.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace StatsKernels
{
using Stats::ReductionType;

__global__ void totalMomentumEnergy(PVview view, ReductionType *momentum, ReductionType *energy, float* maxvel)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float3 vel, myMomentum;
    float myEnergy = 0.f, myMaxIvelI;
    vel = myMomentum = make_float3(0.f);

    if (tid < view.size)
    {
        vel        = make_float3(view.readVelocity(tid));
        myMomentum = vel * view.mass;
        myEnergy   = dot(vel, vel) * view.mass * 0.5f;
    }
    
    myMomentum = warpReduce(myMomentum, [](float a, float b) { return a+b; });
    myEnergy   = warpReduce(myEnergy,   [](float a, float b) { return a+b; });
    
    myMaxIvelI = warpReduce(length(vel), [](float a, float b) { return max(a, b); });

    if (laneId() == 0)
    {
        atomicAdd(momentum+0, (ReductionType)myMomentum.x);
        atomicAdd(momentum+1, (ReductionType)myMomentum.y);
        atomicAdd(momentum+2, (ReductionType)myMomentum.z);
        atomicAdd(energy,     (ReductionType)myEnergy);

        atomicMax((int*)maxvel, __float_as_int(myMaxIvelI));
    }
}
} // namespace StatsKernels
    
SimulationStats::SimulationStats(const MirState *state, std::string name, int fetchEvery) :
    SimulationPlugin(state, name),
    fetchEvery(fetchEvery)
{
    timer.start();
}

SimulationStats::~SimulationStats() = default;

void SimulationStats::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);
    pvs = simulation->getParticleVectors();
}

void SimulationStats::afterIntegration(cudaStream_t stream)
{
    if (!isTimeEvery(state, fetchEvery)) return;

    momentum.clear(stream);
    energy  .clear(stream);
    maxvel  .clear(stream);

    nparticles = 0;
    for (auto& pv : pvs)
    {
        PVview view(pv, pv->local());

        SAFE_KERNEL_LAUNCH(
                StatsKernels::totalMomentumEnergy,
                getNblocks(view.size, 128), 128, 0, stream,
                view, momentum.devPtr(), energy.devPtr(), maxvel.devPtr() );

        nparticles += view.size;
    }

    momentum.downloadFromDevice(stream, ContainersSynch::Asynch);
    energy  .downloadFromDevice(stream, ContainersSynch::Asynch);
    maxvel  .downloadFromDevice(stream);

    needToDump = true;
}

void SimulationStats::serializeAndSend(__UNUSED cudaStream_t stream)
{
    if (needToDump)
    {
        float tm = timer.elapsedAndReset() / (state->currentStep < fetchEvery ? 1.0f : fetchEvery);
        waitPrevSend();
        SimpleSerializer::serialize(sendBuffer, tm, state->currentTime, state->currentStep, nparticles, momentum, energy, maxvel);
        send(sendBuffer);
        needToDump = false;
    }
}

PostprocessStats::PostprocessStats(std::string name, std::string filename) :
        PostprocessPlugin(name)
{
    if (std::is_same<Stats::ReductionType, float>::value)
        mpiReductionType = MPI_FLOAT;
    else if (std::is_same<Stats::ReductionType, double>::value)
        mpiReductionType = MPI_DOUBLE;
    else
        die("Incompatible type");

    if (std::is_same<Stats::CountType, unsigned long long>::value)
        mpiCountType = MPI_UNSIGNED_LONG_LONG;
    else
        die("Incompatible type");    
    
    if (filename != "")
    {
        auto status = fdump.open(filename, "w");
        if (status != FileWrapper::Status::Success)
            die("Could not open file '%s'", filename.c_str());

        fprintf(fdump.get(), "# time  kBT  vx vy vz  max(abs(v)) num_particles simulation_time_per_step(ms)\n");
    }
}

void PostprocessStats::deserialize(__UNUSED MPI_Status& stat)
{
    MirState::TimeType currentTime;
    MirState::StepType currentTimeStep;
    float realTime;
    Stats::CountType nparticles, maxNparticles, minNparticles;

    std::vector<Stats::ReductionType> momentum, energy;
    std::vector<float> maxvel;

    SimpleSerializer::deserialize(data, realTime, currentTime, currentTimeStep, nparticles, momentum, energy, maxvel);

    MPI_Check( MPI_Reduce(&nparticles, &minNparticles, 1, mpiCountType, MPI_MIN, 0, comm) );
    MPI_Check( MPI_Reduce(&nparticles, &maxNparticles, 1, mpiCountType, MPI_MAX, 0, comm) );
    
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &nparticles,     &nparticles,     1, mpiCountType,     MPI_SUM, 0, comm) );
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : energy.data(),   energy.data(),   1, mpiReductionType, MPI_SUM, 0, comm) );
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : momentum.data(), momentum.data(), 3, mpiReductionType, MPI_SUM, 0, comm) );

    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : maxvel.data(),   maxvel.data(),   1, MPI_FLOAT,        MPI_MAX, 0, comm) );

    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &realTime,       &realTime,       1, MPI_FLOAT,        MPI_MAX, 0, comm) );

    if (rank == 0)
    {
        double invNparticles = nparticles > 0 ? 1.0 / nparticles : 0.0;
        momentum[0] *= invNparticles;
        momentum[1] *= invNparticles;
        momentum[2] *= invNparticles;
        const Stats::ReductionType temperature = energy[0] * invNparticles * (2.0/3.0);

        printf("Stats at timestep %lld (simulation time %f):\n", currentTimeStep, currentTime);
        printf("\tOne timestep takes %.2f ms", realTime);
        printf("\tNumber of particles (total, min/proc, max/proc): %llu,  %llu,  %llu\n", nparticles, minNparticles, maxNparticles);
        printf("\tAverage momentum: [%e %e %e]\n", momentum[0], momentum[1], momentum[2]);
        printf("\tMax velocity magnitude: %f\n", maxvel[0]);
        printf("\tTemperature: %.4f\n\n", temperature);

        if (fdump.get())
        {
            fprintf(fdump.get(), "%g %g %g %g %g %g %llu %g\n", currentTime,
                    temperature, momentum[0], momentum[1], momentum[2],
                    maxvel[0], nparticles, realTime);
            fflush(fdump.get());
        }
    }
}


