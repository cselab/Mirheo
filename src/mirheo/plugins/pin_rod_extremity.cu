// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "pin_rod_extremity.h"

#include <mirheo/core/pvs/rod_vector.h>
#include <mirheo/core/pvs/views/rv.h>
#include <mirheo/core/simulation.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo
{

namespace pin_rod_extremity_kernels
{

__device__ inline real3 fetchPosition(const RVview& view, int i)
{
    Real3_int ri(view.readPosition(i));
    return ri.v;
}

__device__ inline real3 fetchForce(const RVview& view, int i)
{
    Real3_int f(view.forces[i]);
    return f.v;
}

__global__ void alignMaterialFrame(RVview view, int segmentId, real k, real3 target)
{
    int rodId = threadIdx.x + blockIdx.x * blockDim.x;
    if (rodId >= view.nObjects) return;

    int start = rodId * view.objSize + segmentId * 5;

    auto r0 = fetchPosition(view, start + 0);
    auto u0 = fetchPosition(view, start + 1);
    auto u1 = fetchPosition(view, start + 2);
    auto r1 = fetchPosition(view, start + 5);

    auto t  = normalize(r1 - r0);

    real3 du = u1 - u0;
    du = du - t * dot(du, t);

    real3 a = normalize(target - t * dot(target, t));

    real inv_du = math::rsqrt(dot(du, du));

    real3 fu0 = k * inv_du * (a - (inv_du * inv_du * dot(du, a)) * du);

    atomicAdd(&view.forces[start+1],  fu0);
    atomicAdd(&view.forces[start+2], -fu0);
}

} // namespace pin_rod_extremity_kernels

PinRodExtremityPlugin::PinRodExtremityPlugin(const MirState *state, std::string name, std::string rvName,
                                             int segmentId, real fmagn, real3 targetDirection) :
    SimulationPlugin(state, name),
    rvName_(rvName),
    segmentId_(segmentId),
    fmagn_(fmagn),
    targetDirection_(targetDirection)
{}

void PinRodExtremityPlugin::setup(Simulation *simulation, const MPI_Comm& comm, const MPI_Comm& interComm)
{
    SimulationPlugin::setup(simulation, comm, interComm);

    auto ov = simulation->getOVbyNameOrDie(rvName_);

    rv_ = dynamic_cast<RodVector*>(ov);

    if (rv_ == nullptr)
        die("Plugin '%s' must be used with a rod vector; given PV '%s'",
            getCName(), rvName_.c_str());

    if (segmentId_ < 0 || segmentId_ >= rv_->local()->getNumSegmentsPerRod())
        die("Invalid segment id in plugin '%s'", getCName());
}

void PinRodExtremityPlugin::beforeIntegration(cudaStream_t stream)
{
    RVview view(rv_, rv_->local());

    const int nthreads = 32;
    const int nblocks = getNblocks(view.nObjects, nthreads);

    SAFE_KERNEL_LAUNCH(pin_rod_extremity_kernels::alignMaterialFrame,
                       nblocks, nthreads, 0, stream,
                       view, segmentId_, fmagn_, targetDirection_ );
}

} // namespace mirheo
