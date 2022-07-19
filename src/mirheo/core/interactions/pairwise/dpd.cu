// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "dpd.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

PairwiseDPDInteraction::PairwiseDPDInteraction(const MirState *state,
                                               const std::string& name,
                                               real rc,
                                               DPDParams params,
                                               std::optional<real> stressPeriod)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)
{
    if (stressPeriod)
    {
        pairWithStress_ = PairwiseStressWrapper<PairwiseDPD>(rc, params);
        stressManager_ = StressManager(*stressPeriod);
    }
}

void PairwiseDPDInteraction::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    if (stressManager_)
    {
        pv1->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);
        pv2->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);

        cl1->requireExtraDataPerParticle <Stress> (channel_names::stresses);
        cl2->requireExtraDataPerParticle <Stress> (channel_names::stresses);
    }
}

void PairwiseDPDInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
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

void PairwiseDPDInteraction::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
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

std::vector<Interaction::InteractionChannel> PairwiseDPDInteraction::getOutputChannels() const
{
    std::vector<InteractionChannel> channels = {{channel_names::forces, alwaysActive}};

    if (stressManager_)
    {
        channels.push_back(stressManager_->getStressPredicate(getState()));
    }

    return channels;
}

} // namespace mirheo
