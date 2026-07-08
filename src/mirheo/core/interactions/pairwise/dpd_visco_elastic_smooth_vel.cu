// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "dpd_visco_elastic_smooth_vel.h"

#include "symmetric_pairwise_helpers.h"

#include <mirheo/core/pvs/views/pv_with_pol_chain_smooth_velocity.h>
#include <mirheo/core/utils/cuda_rng.h>
#include <mirheo/core/utils/kernel_launch.h>
#include <mirheo/core/utils/cuda_common.h>

namespace mirheo {

namespace visco_elastic_smooth_vel_dpd_kernels {

__global__ void chainFluctuationRelaxation(PVviewWithPolChainVector view, real sigma, real k, real seed)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= view.size)
        return;

    const real3 Q = view.Q[i];

    const real xix = Saru::mean0var1(seed, 45*i+47, 96*i-65789);
    const real xiy = Saru::mean0var1(xix, 47*i+43, 16*i-6578);
    const real xiz = Saru::mean0var1(xiy, 49*i+41, 196*i+657);

    real3 dQdt = sigma * real3{xix, xiy, xiz};
    dQdt -= k * Q;

    atomicAdd(view.dQdt + i, dQdt);
}

} // namespace visco_elastic_smooth_vel_dpd_kernels

PairwiseViscoElasticSmoothVelDPDInteraction::PairwiseViscoElasticSmoothVelDPDInteraction(const MirState *state,
                                                                                         const std::string& name,
                                                                                         real rc,
                                                                                         ViscoElasticDPDParams params)
    : BasePairwiseInteraction(state, name, rc)
    , params_(params)
    , pair_(rc, params)
{}

void PairwiseViscoElasticSmoothVelDPDInteraction::setPrerequisites(ParticleVector *pv1,
                                                                   ParticleVector *pv2,
                                                                   CellList *cl1,
                                                                   CellList *cl2)
{
    pv1->requireDataPerParticle <real3> (channel_names::polChainVectors, DataManager::PersistenceMode::Active);
    pv2->requireDataPerParticle <real3> (channel_names::polChainVectors, DataManager::PersistenceMode::Active);

    pv1->requireDataPerParticle <real3> (channel_names::derChainVectors, DataManager::PersistenceMode::None);
    pv2->requireDataPerParticle <real3> (channel_names::derChainVectors, DataManager::PersistenceMode::None);

    pv1->requireDataPerParticle <real4> (channel_names::smoothVelocities, DataManager::PersistenceMode::Active);
    pv2->requireDataPerParticle <real4> (channel_names::smoothVelocities, DataManager::PersistenceMode::Active);

    cl1->requireExtraDataPerParticle <real3> (channel_names::polChainVectors);
    cl2->requireExtraDataPerParticle <real3> (channel_names::polChainVectors);

    cl1->requireExtraDataPerParticle <real3> (channel_names::derChainVectors);
    cl2->requireExtraDataPerParticle <real3> (channel_names::derChainVectors);

    cl1->requireExtraDataPerParticle <real4> (channel_names::smoothVelocities);
    cl2->requireExtraDataPerParticle <real4> (channel_names::smoothVelocities);
}

void PairwiseViscoElasticSmoothVelDPDInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
                                                        CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    // single particle part; we skip the cases pv1 != pv2 to avoid accounting it multiple times.
    // We assume here that pv1 always interacts with itself exactly once with that interaction.
    if (pv1 == pv2)
    {
        const real dt = getState()->getDt();

        const real H = params_.H;
        const real sigma = std::sqrt(4.0_r * params_.kBTC / (dt * params_.zeta));
        const real k = 2.0_r * H / params_.zeta;

        const auto seed = stepGen_.generate(getState());

        PVviewWithPolChainVector view(pv1, pv1->local());

        constexpr int nthreads = 128;
        const int nblocks = getNblocks(view.size, nthreads);

        SAFE_KERNEL_LAUNCH(
            visco_elastic_smooth_vel_dpd_kernels::chainFluctuationRelaxation,
            nblocks, nthreads, 0, stream,
            view, sigma, k, seed);
    }

    symmetric_pairwise_helpers::computeLocalInteractions(getState(), pair_, pv1, pv2, cl1, cl2, stream);
}

void PairwiseViscoElasticSmoothVelDPDInteraction::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
                                                       CellList *cl2, cudaStream_t stream)
{
    symmetric_pairwise_helpers::computeHaloInteractions(getState(), pair_, pv1, pv2, cl1, cl2, stream);
}

std::vector<Interaction::InteractionChannel> PairwiseViscoElasticSmoothVelDPDInteraction::getInputChannels() const
{
    return {{channel_names::polChainVectors, alwaysActive},
            {channel_names::smoothVelocities, alwaysActive}};
}

std::vector<Interaction::InteractionChannel> PairwiseViscoElasticSmoothVelDPDInteraction::getOutputChannels() const
{
    return {{channel_names::forces, alwaysActive},
            {channel_names::derChainVectors, alwaysActive}};
}


} // namespace mirheo
