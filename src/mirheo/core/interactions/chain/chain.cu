// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "chain.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/chain_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace chain_forces_kernels {


__device__ inline real3 fspringFENE(real3 r0, real3 r1, real ks, real rmax2)
{
    auto dr = r1 - r0;
    auto r2 = dot(dr, dr);

    auto fmagn = ks * rmax2 / math::max(rmax2 - r2, 1e-4_r);

    return fmagn * dr;
}

__global__ void computeFENESpringForces(OVview view, real ks, real rmax2)
{
    const int numSpringsPerChain = view.objSize - 1;

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int chainId   = i / numSpringsPerChain;
    const int segmentId = i % numSpringsPerChain;
    const int start = view.objSize * chainId + segmentId;

    if (chainId   >= view.nObjects )     return;
    if (segmentId >= numSpringsPerChain) return;

    const real3 r0 = make_real3(view.readPosition(start + 0));
    const real3 r1 = make_real3(view.readPosition(start + 1));

    const real3 f01 = fspringFENE(r0, r1, ks, rmax2);

    atomicAdd(view.forces + start + 0,  f01);
    atomicAdd(view.forces + start + 1, -f01);
}

} // namespace chain_forces_kernels

ChainInteraction::ChainInteraction(const MirState *state, const std::string& name, real ks, real rmax) :
    Interaction(state, name),
    ks_(ks),
    rmax2_(rmax*rmax)
{}

ChainInteraction::~ChainInteraction() = default;

void ChainInteraction::halo(ParticleVector *pv1,
                            __UNUSED ParticleVector *pv2,
                            __UNUSED CellList *cl1,
                            __UNUSED CellList *cl2,
                            __UNUSED cudaStream_t stream)
{
    debug("Not computing internal FENE forces between local and halo chains of '%s'", pv1->getCName());
}


void ChainInteraction::local(ParticleVector *pv1,
                             __UNUSED ParticleVector *pv2,
                             __UNUSED CellList *cl1,
                             __UNUSED CellList *cl2,
                             cudaStream_t stream)
{
    auto cv = dynamic_cast<ChainVector*>(pv1);

    if (!cv)
        die("%s expects a chain vector, got '%s'",
            this->getCName(), pv1->getCName());

    debug("Computing internal chain forces for %d chains of '%s'",
          cv->local()->getNumObjects(), cv->getCName());


    OVview view(cv, cv->local());


    const int nthreads = 128;
    const int nblocks  = getNblocks(view.nObjects * view.objSize, nthreads);

    SAFE_KERNEL_LAUNCH(chain_forces_kernels::computeFENESpringForces,
                       nblocks, nthreads, 0, stream,
                       view, ks_, rmax2_);
}


bool ChainInteraction::isSelfObjectInteraction() const
{
    return true;
}


} // namespace mirheo
