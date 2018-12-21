#include "temperaturize.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>

#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>


__global__ void applyTemperature(PVview view, float kbT, float seed1, float seed2, bool keepVelocity)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    float2 rand1 = Saru::normal2(seed1, threadIdx.x, blockIdx.x);
    float2 rand2 = Saru::normal2(seed2, threadIdx.x, blockIdx.x);

    float3 vel = sqrtf(kbT * view.invMass) * make_float3(rand1.x, rand1.y, rand2.x);

    Float3_int u(view.particles[2*gid+1]);
    if (keepVelocity) u.v += vel;
    else              u.v  = vel;

    view.particles[2*gid+1] = u.toFloat4();
}

TemperaturizePlugin::TemperaturizePlugin(const YmrState *state, std::string name, std::string pvName, float kbT, bool keepVelocity) :
    SimulationPlugin(state, name),
    pvName(pvName),
    kbT(kbT),
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
            view, kbT, drand48(), drand48(), keepVelocity );
}

