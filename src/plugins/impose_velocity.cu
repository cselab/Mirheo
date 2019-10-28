#include "impose_velocity.h"
#include "utils/time_stamp.h"

#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>
#include <core/utils/kernel_launch.h>

namespace ImposeVelocityKernels
{
__global__ void addVelocity(PVview view, DomainInfo domain, real3 low, real3 high, real3 extraVel)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    Particle p(view.readParticle(gid));
    real3 gr = domain.local2global(p.r);

    if (low.x <= gr.x && gr.x <= high.x &&
        low.y <= gr.y && gr.y <= high.y &&
        low.z <= gr.z && gr.z <= high.z)
    {
        p.u += extraVel;
        view.writeVelocity(gid, p.u2Real4());
    }
}

__global__ void averageVelocity(PVview view, DomainInfo domain, real3 low, real3 high, double3 *totVel, int *nSamples)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    Particle p;

    p.u = make_real3(0._r);

    if (gid < view.size) {

        p = view.readParticle(gid);
        real3 gr = domain.local2global(p.r);

        if (low.x <= gr.x && gr.x <= high.x &&
            low.y <= gr.y && gr.y <= high.y &&
            low.z <= gr.z && gr.z <= high.z)
        {
            atomicAggInc(nSamples);
        }
        else
        {
            p.u = make_real3(0.0_r);
        }
    }
    
    real3 u = warpReduce(p.u, [](real a, real b) { return a+b; });
    if (laneId() == 0 && dot(u, u) > 1e-8_r)
    {
        atomicAdd(&totVel->x, (double)u.x);
        atomicAdd(&totVel->y, (double)u.y);
        atomicAdd(&totVel->z, (double)u.z);
    }
}
} // namespace ImposeVelocityKernels

ImposeVelocityPlugin::ImposeVelocityPlugin(const MirState *state, std::string name, std::vector<std::string> pvNames,
                                           real3 low, real3 high, real3 targetVel, int every) :
    SimulationPlugin(state, name),
    pvNames(pvNames),
    low(low),
    high(high),
    targetVel(targetVel),
    every(every)
{}

void ImposeVelocityPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    for (auto& nm : pvNames)
        pvs.push_back(simulation->getPVbyNameOrDie(nm));
}

void ImposeVelocityPlugin::afterIntegration(cudaStream_t stream)
{
    if (isTimeEvery(state, every))
    {
        const int nthreads = 128;

        totVel.clearDevice(stream);
        nSamples.clearDevice(stream);
        
        for (auto& pv : pvs)
            SAFE_KERNEL_LAUNCH(
                    ImposeVelocityKernels::averageVelocity,
                    getNblocks(pv->local()->size(), nthreads), nthreads, 0, stream,
                    PVview(pv, pv->local()), state->domain, low, high, totVel.devPtr(), nSamples.devPtr() );

        totVel.downloadFromDevice(stream, ContainersSynch::Asynch);
        nSamples.downloadFromDevice(stream);

        real3 avgVel = make_real3(totVel[0].x / nSamples[0], totVel[0].y / nSamples[0], totVel[0].z / nSamples[0]);

        debug("Current mean velocity measured by plugin '%s' is [%f %f %f]; as of %d particles",
              name.c_str(), avgVel.x, avgVel.y, avgVel.z, nSamples[0]);

        for (auto& pv : pvs)
            SAFE_KERNEL_LAUNCH(
                    ImposeVelocityKernels::addVelocity,
                    getNblocks(pv->local()->size(), nthreads), nthreads, 0, stream,
                    PVview(pv, pv->local()), state->domain, low, high, targetVel - avgVel);
    }
}

void ImposeVelocityPlugin::setTargetVelocity(real3 v)
{
    info("Changing target velocity from [%f %f %f] to [%f %f %f]",
         targetVel.x, targetVel.y, targetVel.z,
         v.x, v.y, v.z);
    
    targetVel = v;
}

