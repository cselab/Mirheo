#include "exchange_pvs_flux_plane.h"

#include <core/utils/kernel_launch.h>
#include <core/pvs/particle_vector.h>
#include <core/pvs/views/pv.h>
#include <core/simulation.h>

#include <core/utils/cuda_common.h>

namespace exchange_pvs_flux_plane_kernels {

__device__ inline bool hasCrossedPlane(DomainInfo domain, float3 pos, float4 plane)
{
    pos = domain.local2global(pos);
    return plane.x * pos.x + plane.y * pos.y + plane.z * pos.z + plane.w >= 0.f;
}

// TODO use cell lists
__global__ void countParticles(DomainInfo domain, PVview view1, float4 plane, int *numberCrossed)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view1.size) return;

    Particle p;
    p.readCoordinate(view1.particles, pid);
    if (hasCrossedPlane(domain, p.r, plane))
        atomicAdd(numberCrossed, 1);
}

__global__ void moveParticles(DomainInfo domain, PVview view1, PVview view2, float4 plane, int oldsize2, int *numberCrossed)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= view1.size) return;

    Particle p;
    p.readCoordinate(view1.particles, pid);

    if (hasCrossedPlane(domain, p.r, plane)) {
        p.readVelocity(view1.particles, pid);
        int dst = atomicAdd(numberCrossed, 1);
        dst += oldsize2;

        p.write2Float4(view2.particles, dst);

        p.mark();
        p.write2Float4(view1.particles, pid);        
    }
}

}


ExchangePVSFluxPlanePlugin::ExchangePVSFluxPlanePlugin(std::string name, const YmrState *state, std::string pv1Name, std::string pv2Name, float4 plane) :
    SimulationPlugin(name, state), pv1Name(pv1Name), pv2Name(pv2Name), plane(plane), numberCrossedParticles(1)
{}


void ExchangePVSFluxPlanePlugin::setup(Simulation* simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    pv1 = simulation->getPVbyNameOrDie(pv1Name);
    pv2 = simulation->getPVbyNameOrDie(pv2Name);
}

void ExchangePVSFluxPlanePlugin::beforeParticleDistribution(cudaStream_t stream)
{
    DomainInfo domain = state->domain;
    PVview view1(pv1, pv1->local());
    PVview view2(pv2, pv2->local());
    const int nthreads = 128;

    numberCrossedParticles.clear(stream);
    
    SAFE_KERNEL_LAUNCH(
            exchange_pvs_flux_plane_kernels::countParticles,
            getNblocks(view1.size, nthreads), nthreads, 0, stream,
            domain, view1, plane, numberCrossedParticles.devPtr() );

    numberCrossedParticles.downloadFromDevice(stream, ContainersSynch::Synch);

    const int old_size2 = view2.size;
    const int new_size2 = old_size2 + numberCrossedParticles.hostPtr()[0];

    pv2->local()->resize(new_size2, stream);
    numberCrossedParticles.clear(stream);

    view2 = PVview(pv2, pv2->local());

    SAFE_KERNEL_LAUNCH(
                       exchange_pvs_flux_plane_kernels::moveParticles,
                       getNblocks(view1.size, nthreads), nthreads, 0, stream,
                       domain, view1, view2, plane, old_size2, numberCrossedParticles.devPtr() );

}

