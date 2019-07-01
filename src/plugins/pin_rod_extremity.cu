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

__global__ void alignMaterialFrame(RVview view, int segmentId, float k, float3 target)
{
    int rodId = threadIdx.x + blockIdx.x * blockDim.x;
    if (rodId >= view.nObjects) return;
    
    int start = rodId * view.objSize + segmentId * 5;

    auto r0 = fetchPosition(view, start + 0);
    auto u0 = fetchPosition(view, start + 1);
    auto u1 = fetchPosition(view, start + 2);
    auto r1 = fetchPosition(view, start + 5);

    auto t  = normalize(r1 - r0);

    float3 du = u1 - u0;
    du = du - t * dot(du, t);
    
    float3 a = normalize(target - t * dot(target, t));

    float inv_du = rsqrt(dot(du, du));

    float3 fu0 = k * inv_du * (a - (inv_du * inv_du * dot(du, a)) * du);
    
    atomicAdd(&view.forces[start+1],  fu0);
    atomicAdd(&view.forces[start+2], -fu0);
}

} // namespace PinRodExtremityKernels

PinRodExtremityPlugin::PinRodExtremityPlugin(const MirState *state, std::string name, std::string rvName,
                                             int segmentId, float fmagn, float3 targetDirection) :
    SimulationPlugin(state, name),
    rvName(rvName),
    segmentId(segmentId),
    fmagn(fmagn),
    targetDirection(targetDirection)
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
    
    SAFE_KERNEL_LAUNCH(PinRodExtremityKernels::alignMaterialFrame,
                       nblocks, nthreads, 0, stream,
                       view, segmentId, fmagn, targetDirection );
}

