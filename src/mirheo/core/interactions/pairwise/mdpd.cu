// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "mdpd.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

PairwiseMDPDInteraction::PairwiseMDPDInteraction(const MirState *state,
                                                 const std::string& name,
                                                 real rc,
                                                 MDPDParams params,
                                                 std::optional<real> stressPeriod)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)

{
    if (stressPeriod)
    {
        pairWithStress_ = PairwiseStressWrapper<PairwiseMDPD>(rc, params);
        stressManager_ = StressManager(*stressPeriod);
    }
}

void PairwiseMDPDInteraction::setPrerequisites(ParticleVector *pv1,
                                               ParticleVector *pv2,
                                               CellList *cl1,
                                               CellList *cl2)
{
    pv1->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<real>(channel_names::densities, DataManager::PersistenceMode::None);

    cl1->requireExtraDataPerParticle<real>(channel_names::densities);
    cl2->requireExtraDataPerParticle<real>(channel_names::densities);

    if (stressManager_)
    {
        pv1->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);
        pv2->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);

        cl1->requireExtraDataPerParticle <Stress> (channel_names::stresses);
        cl2->requireExtraDataPerParticle <Stress> (channel_names::stresses);
    }
}

void PairwiseMDPDInteraction::local(ParticleVector *pv1, ParticleVector *pv2,
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

void PairwiseMDPDInteraction::halo(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1,
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

std::vector<Interaction::InteractionChannel> PairwiseMDPDInteraction::getInputChannels() const
{
    return {{channel_names::densities, Interaction::alwaysActive}};
}

std::vector<Interaction::InteractionChannel> PairwiseMDPDInteraction::getOutputChannels() const
{
    std::vector<InteractionChannel> channels = {{channel_names::forces, alwaysActive}};

    if (stressManager_)
    {
        channels.push_back(stressManager_->getStressPredicate(getState()));
    }

    return channels;
}

} // namespace mirheo
