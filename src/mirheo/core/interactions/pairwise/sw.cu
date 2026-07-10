// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "sw.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

PairwiseSWInteraction::PairwiseSWInteraction(const MirState *state,
                                             const std::string& name,
                                             real rc,
                                             SW2Params params,
                                             std::optional<real> stressPeriod)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)
{
    if (stressPeriod)
    {
        pairWithStress_ = PairwiseStressWrapper<PairwiseSW>(rc, params);
        stressManager_ = StressManager(*stressPeriod);
    }
}

void PairwiseSWInteraction::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    if (stressManager_)
    {
        pv1->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);
        pv2->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);

        cl1->requireExtraDataPerParticle <Stress> (channel_names::stresses);
        cl2->requireExtraDataPerParticle <Stress> (channel_names::stresses);
    }
}

void PairwiseSWInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
                                  CellList *cl1, CellList *cl2, cudaStream_t stream)
{
    if (stressManager_)
    {
        stressManager_->computeLocalInteractions(getState(),
                                                 pair_, *pairWithStress_,
                                                 pv1, pv2, cl1, cl2, stream);
    }
    else
    {
        symmetric_pairwise_helpers::computeLocalInteractions(getState(), pair_, pv1, pv2, cl1, cl2, stream);
    }
}

void PairwiseSWInteraction::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
                                 CellList *cl2, cudaStream_t stream)
{
    if (stressManager_)
    {
        stressManager_->computeHaloInteractions(getState(),
                                                pair_, *pairWithStress_,
                                                pv1, pv2, cl1, cl2, stream);
    }
    else
    {
        symmetric_pairwise_helpers::computeHaloInteractions(getState(), pair_, pv1, pv2, cl1, cl2, stream);
    }
}

std::vector<Interaction::InteractionChannel> PairwiseSWInteraction::getOutputChannels() const
{
    std::vector<InteractionChannel> channels = {{channel_names::forces, alwaysActive}};

    if (stressManager_)
    {
        channels.push_back(stressManager_->getStressPredicate(getState()));
    }

    return channels;
}

} // namespace mirheo
