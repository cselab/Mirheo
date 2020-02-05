#include "impose_velocity.h"
#include "utils/time_stamp.h"

#include <mirheo/core/pvs/particle_vector.h>
#include <mirheo/core/pvs/views/pv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

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
    pvNames_(pvNames),
    low_(low),
    high_(high),
    targetVel_(targetVel),
    every_(every)
{}

void ImposeVelocityPlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    for (auto& nm : pvNames_)
        pvs_.push_back(simulation->getPVbyNameOrDie(nm));
}

void ImposeVelocityPlugin::afterIntegration(cudaStream_t stream)
{
    if (isTimeEvery(getState(), every_))
    {
        const int nthreads = 128;

        totVel_.clearDevice(stream);
        nSamples_.clearDevice(stream);
        
        for (auto& pv : pvs_)
        {
            SAFE_KERNEL_LAUNCH(
                    ImposeVelocityKernels::averageVelocity,
                    getNblocks(pv->local()->size(), nthreads), nthreads, 0, stream,
                    PVview(pv, pv->local()), getState()->domain, low_, high_, totVel_.devPtr(), nSamples_.devPtr() );
        }
        
        totVel_.downloadFromDevice(stream, ContainersSynch::Asynch);
        nSamples_.downloadFromDevice(stream);

        const real3 avgVel = make_real3(totVel_[0].x / nSamples_[0], totVel_[0].y / nSamples_[0], totVel_[0].z / nSamples_[0]);

        debug("Current mean velocity measured by plugin '%s' is [%f %f %f]; as of %d particles",
              getCName(), avgVel.x, avgVel.y, avgVel.z, nSamples_[0]);

        for (auto& pv : pvs_)
            SAFE_KERNEL_LAUNCH(
                    ImposeVelocityKernels::addVelocity,
                    getNblocks(pv->local()->size(), nthreads), nthreads, 0, stream,
                    PVview(pv, pv->local()), getState()->domain, low_, high_, targetVel_ - avgVel);
    }
}

void ImposeVelocityPlugin::setTargetVelocity(real3 v)
{
    info("Changing target velocity from [%f %f %f] to [%f %f %f]",
         targetVel_.x, targetVel_.y, targetVel_.z,
         v.x, v.y, v.z);
    
    targetVel_ = v;
}

} // namespace mirheo
