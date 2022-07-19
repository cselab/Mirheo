// Copyright 2020 ETH Zurich. All Rights Reserved.
#include "morse.h"
#include "symmetric_pairwise_helpers.h"

namespace mirheo {

template<class Awareness>
PairwiseMorseInteraction<Awareness>::PairwiseMorseInteraction(const MirState *state,
                                                              const std::string& name,
                                                              real rc,
                                                              MorseParams params,
                                                              std::optional<real> stressPeriod)
    : BasePairwiseInteraction(state, name, rc)
    , pair_(rc, params)

{
    if (stressPeriod)
    {
        pairWithStress_ = PairwiseStressWrapper<PairwiseMorse<Awareness>>(rc, params);
        stressManager_ = StressManager(*stressPeriod);
    }
}

template<class Awareness>
void PairwiseMorseInteraction<Awareness>::setPrerequisites(ParticleVector *pv1,
                                                           ParticleVector *pv2,
                                                           CellList *cl1,
                                                           CellList *cl2)
{
    if (stressManager_)
    {
        pv1->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);
        pv2->requireDataPerParticle <Stress> (channel_names::stresses, DataManager::PersistenceMode::None);

        cl1->requireExtraDataPerParticle <Stress> (channel_names::stresses);
        cl2->requireExtraDataPerParticle <Stress> (channel_names::stresses);
    }
}

template<class Awareness>
void PairwiseMorseInteraction<Awareness>::local(ParticleVector *pv1, ParticleVector *pv2,
                                                CellList *cl1, CellList *cl2,
                                                cudaStream_t stream)
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

template<class Awareness>
void PairwiseMorseInteraction<Awareness>::halo(ParticleVector *pv1, ParticleVector *pv2,
                                               CellList *cl1, CellList *cl2,
                                               cudaStream_t stream)
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

template<class Awareness>
std::vector<Interaction::InteractionChannel> PairwiseMorseInteraction<Awareness>::getOutputChannels() const
{
    std::vector<InteractionChannel> channels = {{channel_names::forces, alwaysActive}};

    if (stressManager_)
    {
        channels.push_back(stressManager_->getStressPredicate(getState()));
    }

    return channels;
}




std::unique_ptr<BasePairwiseInteraction>
makePairwiseMorseInteraction(const MirState *state, const std::string& name, real rc,
                             MorseParams params, std::optional<real> stressPeriod)
{
    return std::visit([=](auto awarenessParams) -> std::unique_ptr<BasePairwiseInteraction>
    {
        using AwarenessParamsType = decltype(awarenessParams);
        using Awareness = typename AwarenessParamsType::KernelType;
        return std::make_unique<PairwiseMorseInteraction<Awareness>>(state, name, rc, params, stressPeriod);
    }, params.varAwarenessParams);
}


} // namespace mirheo
