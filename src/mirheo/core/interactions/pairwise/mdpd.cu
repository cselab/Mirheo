// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "mdpd.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

PairwiseMDPDInteraction::PairwiseMDPDInteraction(const MirState *state,
                                                 const std::string& name,
                                                 real rc,
                                                 MDPDParams params)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)

{}

void PairwiseMDPDInteraction::setPrerequisites(ParticleVector *pv1,
                                               ParticleVector *pv2,
                                               CellList *cl1,
                                               CellList *cl2)
{
    pv1->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle<real>(channel_names::densities);
    cl2->requireExtraDataPerParticle<real>(channel_names::densities);
}

void PairwiseMDPDInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
                                    CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeLocalInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}

void PairwiseMDPDInteraction::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
                                   CellList *cl2, cudaStream_t stream)
{
    pair_.setup(pv1->local(), pv2->local(), cl1, cl2, getState());
    symmetric_pairwise_helpers::computeHaloInteractions(pair_, pv1, pv2, cl1, cl2, stream);
}

std::vector<Interaction::InteractionChannel> PairwiseMDPDInteraction::getInputChannels() const
{
    return {{channel_names::densities, Interaction::alwaysActive}};
}


} // namespace mirheo
