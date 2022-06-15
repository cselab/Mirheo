// Copyright 2022 ETH Zurich. All Rights Reserved.
#include "chain.h"

#include <mirheo/core/logger.h>
#include <mirheo/core/pvs/chain_vector.h>
#include <mirheo/core/pvs/views/ov.h>
#include <mirheo/core/celllist.h>
#include <mirheo/core/pvs/views/pv_with_stresses.h>
#include <mirheo/core/utils/cuda_common.h>
#include <mirheo/core/utils/kernel_launch.h>

namespace mirheo {
namespace chain_forces_kernels {


__device__ inline real3 fspringFENE(real3 dr, real ks, real rmax2)
{
    const real r2 = dot(dr, dr);
    const real  fmagn = ks * rmax2 / math::max(rmax2 - r2, 1e-4_r);
    return fmagn * dr;
}

template<class ViewType>
__global__ void computeFENESpringForces(ViewType view, real ks, real rmax2)
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

    const real3 dr = r1 - r0;
    const real3 f01 = fspringFENE(dr, ks, rmax2);

    atomicAdd(view.forces + start + 0,  f01);
    atomicAdd(view.forces + start + 1, -f01);

    if constexpr (std::is_same_v<ViewType, PVviewWithStresses<OVview>>) {
        const real Sxx = 0.5_r * dr.x * f01.x;
        const real Sxy = 0.5_r * dr.x * f01.y;
        const real Sxz = 0.5_r * dr.x * f01.z;
        const real Syy = 0.5_r * dr.y * f01.y;
        const real Syz = 0.5_r * dr.y * f01.z;
        const real Szz = 0.5_r * dr.z * f01.z;

        atomicAdd(&view.stresses[start + 0].xx, Sxx);
        atomicAdd(&view.stresses[start + 0].xy, Sxy);
        atomicAdd(&view.stresses[start + 0].xz, Sxz);
        atomicAdd(&view.stresses[start + 0].yy, Syy);
        atomicAdd(&view.stresses[start + 0].yz, Syz);
        atomicAdd(&view.stresses[start + 0].zz, Szz);

        atomicAdd(&view.stresses[start + 1].xx, Sxx);
        atomicAdd(&view.stresses[start + 1].xy, Sxy);
        atomicAdd(&view.stresses[start + 1].xz, Sxz);
        atomicAdd(&view.stresses[start + 1].yy, Syy);
        atomicAdd(&view.stresses[start + 1].yz, Syz);
        atomicAdd(&view.stresses[start + 1].zz, Szz);
    }
}

} // namespace chain_forces_kernels

ChainInteraction::ChainInteraction(const MirState *state, const std::string& name, real ks, real rmax,
                                   std::optional<real> stressPeriod) :

    Interaction(state, name),
    ks_(ks),
    rmax2_(rmax*rmax),
    stressPeriod_(std::move(stressPeriod))
{
    if (stressPeriod_)
        lastStressTime_ = -1e6_r;
}

void ChainInteraction::setPrerequisites(ParticleVector *pv1,
                                        __UNUSED ParticleVector *pv2,
                                        CellList *cl1,
                                        __UNUSED CellList *cl2)
{
    if (stressPeriod_)
    {
        pv1->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);
        cl1->requireExtraDataPerParticle <Stress> (channel_names::stresses);
    }
}

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


    if (_isStressTime())
    {
        PVviewWithStresses<OVview> view(cv, cv->local());

        const int nthreads = 128;
        const int nblocks  = getNblocks(view.nObjects * view.objSize, nthreads);

        SAFE_KERNEL_LAUNCH(chain_forces_kernels::computeFENESpringForces,
                           nblocks, nthreads, 0, stream,
                           view, ks_, rmax2_);

        lastStressTime_ = getState()->currentTime;
    }
    else
    {
        OVview view(cv, cv->local());

        const int nthreads = 128;
        const int nblocks  = getNblocks(view.nObjects * view.objSize, nthreads);

        SAFE_KERNEL_LAUNCH(chain_forces_kernels::computeFENESpringForces,
                           nblocks, nthreads, 0, stream,
                           view, ks_, rmax2_);
    }
}


bool ChainInteraction::isSelfObjectInteraction() const
{
    return true;
}

std::vector<Interaction::InteractionChannel> ChainInteraction::getOutputChannels() const
{
    auto channels = Interaction::getOutputChannels();

    if (stressPeriod_)
    {
        channels.push_back({channel_names::stresses,
                            [this]() {return this->_isStressTime();}});
    }

    return channels;
}


bool ChainInteraction::_isStressTime() const
{
    if (!stressPeriod_)
        return false;

    const real t = getState()->currentTime;
    const real last = *lastStressTime_;
    const real period = *stressPeriod_;

    return last + period <= t || last == t;
}

} // namespace mirheo
