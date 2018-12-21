#include "impose_velocity.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>

#include <core/utils/cuda_common.h>
#include <core/utils/cuda_rng.h>


__global__ void addVelocity(PVview view, DomainInfo domain, float3 low, float3 high, float3 extraVel)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    Particle p(view.particles, gid);
    float3 gr = domain.local2global(p.r);

    if (low.x <= gr.x && gr.x <= high.x &&
        low.y <= gr.y && gr.y <= high.y &&
        low.z <= gr.z && gr.z <= high.z)
    {
        p.u += extraVel;
        view.particles[2*gid+1] = p.u2Float4();
    }
}


__global__ void averageVelocity(PVview view, DomainInfo domain, float3 low, float3 high, double3* totVel, int* nSamples)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= view.size) return;

    Particle p(view.particles, gid);
    float3 gr = domain.local2global(p.r);

    if (low.x <= gr.x && gr.x <= high.x &&
        low.y <= gr.y && gr.y <= high.y &&
        low.z <= gr.z && gr.z <= high.z)
    {
        atomicAggInc(nSamples);
    }
    else
    {
        p.u = make_float3(0.0f);
    }

    float3 u = warpReduce(p.u, [](float a, float b) { return a+b; });
    if (threadIdx.x % 32 == 0 && dot(u, u) > 1e-8f)
    {
        atomicAdd(&totVel[0].x, (double)u.x);
        atomicAdd(&totVel[0].y, (double)u.y);
        atomicAdd(&totVel[0].z, (double)u.z);
    }
}


void ImposeVelocityPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    for (auto& nm : pvNames)
        pvs.push_back(simulation->getPVbyNameOrDie(nm));
}

void ImposeVelocityPlugin::afterIntegration(cudaStream_t stream)
{
    if (currentTimeStep % every == 0)
    {
        const int nthreads = 128;

        totVel.clearDevice(stream);
        nSamples.clearDevice(stream);
        
        for (auto& pv : pvs)
            SAFE_KERNEL_LAUNCH(
                    averageVelocity,
                    getNblocks(pv->local()->size(), nthreads), nthreads, 0, stream,
                    PVview(pv, pv->local()), state->domain, low, high, totVel.devPtr(), nSamples.devPtr() );

        totVel.downloadFromDevice(stream, ContainersSynch::Asynch);
        nSamples.downloadFromDevice(stream);

        float3 avgVel = make_float3(totVel[0].x / nSamples[0], totVel[0].y / nSamples[0], totVel[0].z / nSamples[0]);

        debug("Current mean velocity measured by plugin '%s' is [%f %f %f]; as of %d particles",
              name.c_str(), avgVel.x, avgVel.y, avgVel.z, nSamples[0]);

        for (auto& pv : pvs)
            SAFE_KERNEL_LAUNCH(
                    addVelocity,
                    getNblocks(pv->local()->size(), nthreads), nthreads, 0, stream,
                    PVview(pv, pv->local()), state->domain, low, high, targetVel - avgVel);
    }
}

void ImposeVelocityPlugin::setTargetVelocity(PyTypes::float3 v)
{
    info("Changing target velocity from [%f %f %f] to [%f %f %f]",
         targetVel.x, targetVel.y, targetVel.z,
         std::get<0>(v), std::get<1>(v), std::get<2>(v));
    
    targetVel = make_float3(v);
}

