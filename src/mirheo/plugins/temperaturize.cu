#include "temperaturize.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/kernel_launch.h>

__global__ void applyTemperature(PVview view, real kBT, real seed1, real seed2, bool keepVelocity)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    real2 rand1 = Saru::normal2(seed1, threadIdx.x, blockIdx.x);
    real2 rand2 = Saru::normal2(seed2, threadIdx.x, blockIdx.x);

    real3 vel = math::sqrt(kBT * view.invMass) * make_real3(rand1.x, rand1.y, rand2.x);

    Real3_int u(view.readVelocity(gid));
    if (keepVelocity) u.v += vel;
    else              u.v  = vel;

    view.writeVelocity(gid, u.toReal4());
}

TemperaturizePlugin::TemperaturizePlugin(const MirState *state, std::string name, std::string pvName, real kBT, bool keepVelocity) :
    SimulationPlugin(state, name),
    pvName(pvName),
    kBT(kBT),
    keepVelocity(keepVelocity)
{}


void TemperaturizePlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv = simulation->getPVbyNameOrDie(pvName);
}

void TemperaturizePlugin::beforeForces(cudaStream_t stream)
{
    PVview view(pv, pv->local());
    const int nthreads = 128;

    SAFE_KERNEL_LAUNCH(
            applyTemperature,
            getNblocks(view.size, nthreads), nthreads, 0, stream,
            view, kBT, drand48(), drand48(), keepVelocity );
}

