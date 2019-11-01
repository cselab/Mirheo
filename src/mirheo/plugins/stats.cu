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

namespace mirheo
{

namespace StatsKernels
{
using Stats::ReductionType;

__global__ void totalMomentumEnergy(PVview view, ReductionType *momentum, ReductionType *energy, real* maxvel)
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
        const real tm = timer.elapsedAndReset() / (state->currentStep < fetchEvery ? 1.0_r : fetchEvery);
        waitPrevSend();
        SimpleSerializer::serialize(sendBuffer, tm, state->currentTime, state->currentStep, nparticles, momentum, energy, maxvel);
        send(sendBuffer);
        needToDump = false;
    }
}

PostprocessStats::PostprocessStats(std::string name, std::string filename) :
    PostprocessPlugin(name)
{
    if (filename != "")
    {
        auto status = fdump.open(filename, "w");
        if (status != FileWrapper::Status::Success)
            die("Could not open file '%s'", filename.c_str());

        fprintf(fdump.get(), "# time  kBT  vx vy vz  max(abs(v)) num_particles simulation_time_per_step(ms)\n");
    }
}

void PostprocessStats::deserialize()
{
    MirState::TimeType currentTime;
    MirState::StepType currentTimeStep;
    real realTime;
    Stats::CountType nparticles, maxNparticles, minNparticles;

    std::vector<Stats::ReductionType> momentum, energy;
    std::vector<real> maxvel;

    SimpleSerializer::deserialize(data, realTime, currentTime, currentTimeStep, nparticles, momentum, energy, maxvel);

    MPI_Check( MPI_Reduce(&nparticles, &minNparticles, 1, getMPIIntType<Stats::CountType>(), MPI_MIN, 0, comm) );
    MPI_Check( MPI_Reduce(&nparticles, &maxNparticles, 1, getMPIIntType<Stats::CountType>(), MPI_MAX, 0, comm) );
    
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &nparticles,     &nparticles,     1, getMPIIntType<Stats::CountType>(),       MPI_SUM, 0, comm) );
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : energy.data(),   energy.data(),   1, getMPIFloatType<Stats::ReductionType>(), MPI_SUM, 0, comm) );
    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : momentum.data(), momentum.data(), 3, getMPIFloatType<Stats::ReductionType>(), MPI_SUM, 0, comm) );

    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : maxvel.data(),   maxvel.data(),   1, getMPIFloatType<real>(), MPI_MAX, 0, comm) );

    MPI_Check( MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &realTime,       &realTime,       1, getMPIFloatType<real>(), MPI_MAX, 0, comm) );

    if (rank == 0)
    {
        const double invNparticles = nparticles > 0 ? 1.0 / nparticles : 0.0;
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

} // namespace mirheo
