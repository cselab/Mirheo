#include "pin_rod_extremity.h"

#include <core/pvs/rod_vector.h>
#include <core/pvs/views/rv.h>
#include <core/simulation.h>
#include <core/utils/cuda_common.h>
#include <core/utils/kernel_launch.h>

namespace PinRodExtremityKernels
{

__device__ inline float3 fetchPosition(const RVview& view, int i)
{
    Float3_int ri(view.readPosition(i));
    return ri.v;
}

__device__ inline float3 fetchForce(const RVview& view, int i)
{
    Float3_int f(view.forces[i]);
    return f.v;
}

__global__ void removeTorque(RVview view, int segmentId)
{
    int rodId = threadIdx.x + blockIdx.x * blockDim.x;
    if (rodId >= view.nObjects) return;
    
    int start = rodId * view.objSize + segmentId * 5;

    auto r0 = fetchPosition(view, start + 0);
    auto u0 = fetchPosition(view, start + 1);
    auto u1 = fetchPosition(view, start + 2);
    auto v0 = fetchPosition(view, start + 3);
    auto v1 = fetchPosition(view, start + 4);
    auto r1 = fetchPosition(view, start + 5);

    auto t  = normalize(r1 - r0);

    auto fu0 = fetchForce(view, start + 1);
    auto fu1 = fetchForce(view, start + 2);
    auto fv0 = fetchForce(view, start + 3);
    auto fv1 = fetchForce(view, start + 4);

    float3 T =
        cross(u0 - r0, fu0) +
        cross(u1 - r0, fu1) +
        cross(v0 - r0, fv0) +
        cross(v1 - r0, fv1);

    float3 Tanchor = - dot(T, t) * t;

    float3 du = u1 - u0;
    float3 dv = v1 - v0;

    float3 u = normalize(du - t * dot(du, t));
    float3 v = normalize(dv - t * dot(dv, t));
    
    float3 fu0Anchor = 0.25f * cross(Tanchor, u);
    float3 fv0Anchor = 0.25f * cross(Tanchor, v);

    atomicAdd(&view.forces[start+1],  fu0Anchor);
    atomicAdd(&view.forces[start+2], -fu0Anchor);

    atomicAdd(&view.forces[start+3],  fv0Anchor);
    atomicAdd(&view.forces[start+4], -fv0Anchor);
}

} // namespace PinRodExtremityKernels

PinRodExtremityPlugin::PinRodExtremityPlugin(const YmrState *state, std::string name, std::string rvName, int segmentId) :
    SimulationPlugin(state, name),
    rvName(rvName),
    segmentId(segmentId)
{}

void PinRodExtremityPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    auto ov = simulation->getOVbyNameOrDie(rvName);

    rv = dynamic_cast<RodVector*>(ov);

    if (rv == nullptr)
        die("Plugin '%s' must be used with a rod vector; given PV '%s'",
            name.c_str(), rvName.c_str());

    if (segmentId < 0 || segmentId >= rv->local()->getNumSegmentsPerRod())
        die("Invalid segment id in plugin '%s'");
}

void PinRodExtremityPlugin::beforeIntegration(cudaStream_t stream)
{
    RVview view(rv, rv->local());
    
    const int nthreads = 32;
    const int nblocks = getNblocks(view.nObjects, nthreads);
    
    SAFE_KERNEL_LAUNCH(PinRodExtremityKernels::removeTorque,
                       nblocks, nthreads, 0, stream,
                       view, segmentId );
}

