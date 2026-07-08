// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "lj.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

PairwiseLJInteraction::PairwiseLJInteraction(const MirState *state,
                                             const std::string& name,
                                             real rc,
                                             LJParams params,
                                             std::optional<real> stressPeriod)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)

{
    if (stressPeriod)
    {
        pairWithStress_ = PairwiseStressWrapper<PairwiseLJ>(rc, params);
        stressManager_ = StressManager(*stressPeriod);
    }
}

void PairwiseLJInteraction::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    if (stressManager_)
    {
        pv1->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);
        pv2->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);

        cl1->requireExtraDataPerParticle <Stress> (channel_names::stresses);
        cl2->requireExtraDataPerParticle <Stress> (channel_names::stresses);
    }
}

void PairwiseLJInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
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

void PairwiseLJInteraction::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
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

std::vector<Interaction::InteractionChannel> PairwiseLJInteraction::getOutputChannels() const
{
    std::vector<InteractionChannel> channels = {{channel_names::forces, alwaysActive}};

    if (stressManager_)
    {
        channels.push_back(stressManager_->getStressPredicate(getState()));
    }

    return channels;
}

} // namespace mirheo
