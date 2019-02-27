#include "pairwise_interactions/dpd.h"
#include "pairwise_interactions/lj.h"
#include "pairwise_interactions/lj_object_aware.h"
#include "pairwise_interactions/mdpd.h"
#include "pairwise_with_stress.h"

#include <core/celllist.h>
#include <core/utils/common.h>

template<class PairwiseInteraction>
InteractionPair_withStress<PairwiseInteraction>::InteractionPair_withStress(
    const YmrState *state, std::string name, float rc, float stressPeriod, PairwiseInteraction pair) :

    Interaction(state, name, rc),
    stressPeriod(stressPeriod),
    interaction(state, name, rc, pair),
    interactionWithStress(state, name, rc, PairwiseStressWrapper<PairwiseInteraction>(pair))
{}

template<class PairwiseInteraction>
InteractionPair_withStress<PairwiseInteraction>::~InteractionPair_withStress() = default;

template<class PairwiseInteraction>
void InteractionPair_withStress<PairwiseInteraction>::setPrerequisites(ParticleVector *pv1, ParticleVector *pv2, CellList *cl1, CellList *cl2)
{
    info("Interaction '%s' requires channel '%s' from PVs '%s' and '%s'",
         name.c_str(), ChannelNames::stresses.c_str(), pv1->name.c_str(), pv2->name.c_str());

    pv1->requireDataPerParticle <Stress> (ChannelNames::stresses, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle <Stress> (ChannelNames::stresses, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    auto activePredicate1 = [this, pv1]() {
       float t = state->currentTime;
       return (lastStressTime+stressPeriod <= t || lastStressTime == t)
           && (pv2lastStressTime[pv1] != t);
    };

    auto activePredicate2 = [this, pv2]() {
       float t = state->currentTime;
       return (lastStressTime+stressPeriod <= t || lastStressTime == t)
           && (pv2lastStressTime[pv2] != t);
    };
    
    cl1->requireExtraDataPerParticle <Stress> (ChannelNames::stresses, CellList::ExtraChannelRole::FinalOutput, activePredicate1);
    cl2->requireExtraDataPerParticle <Stress> (ChannelNames::stresses, CellList::ExtraChannelRole::FinalOutput, activePredicate2);

    cl1->setNeededForOutput();
    cl2->setNeededForOutput();

    pv2lastStressTime[pv1] = -1;
    pv2lastStressTime[pv2] = -1;
}

template<class PairwiseInteraction>
std::vector<Interaction::InteractionChannel> InteractionPair_withStress<PairwiseInteraction>::getFinalOutputChannels() const
{
    auto activePredicateStress = [this]() {
       float t = state->currentTime;
       return (lastStressTime+stressPeriod <= t) || (lastStressTime == t);
    };

    return {{ChannelNames::forces, Interaction::alwaysActive},
            {ChannelNames::stresses, activePredicateStress}};
}

template<class PairwiseInteraction>
void InteractionPair_withStress<PairwiseInteraction>::local(
        ParticleVector* pv1, ParticleVector* pv2,
        CellList* cl1, CellList* cl2, cudaStream_t stream)
{
    float t = state->currentTime;
    
    if (lastStressTime+stressPeriod <= t || lastStressTime == t)
    {
        debug("Executing interaction '%s' with stress", name.c_str());

        if (pv2lastStressTime[pv1] != t)
            pv2lastStressTime[pv1] = t;

        if (pv2lastStressTime[pv2] != t)
            pv2lastStressTime[pv2] = t;

        interactionWithStress.local(pv1, pv2, cl1, cl2, stream);
        lastStressTime = t;
    }
    else
        interaction.local(pv1, pv2, cl1, cl2, stream);
}

template<class PairwiseInteraction>
void InteractionPair_withStress<PairwiseInteraction>::halo   (
        ParticleVector *pv1, ParticleVector *pv2,
        CellList *cl1, CellList *cl2,
        cudaStream_t stream)
{
    float t = state->currentTime;
    
    if (lastStressTime+stressPeriod <= t || lastStressTime == t)
    {
        debug("Executing interaction '%s' with stress", name.c_str());

        if (pv2lastStressTime[pv1] != t)
            pv2lastStressTime[pv1] = t;

        if (pv2lastStressTime[pv2] != t)
            pv2lastStressTime[pv2] = t;

        interactionWithStress.halo(pv1, pv2, cl1, cl2, stream);
        lastStressTime = t;
    }
    else
        interaction.halo(pv1, pv2, cl1, cl2, stream);
}

template<class PairwiseInteraction>
void InteractionPair_withStress<PairwiseInteraction>::setSpecificPair(
        std::string pv1name, std::string pv2name, PairwiseInteraction pair)
{
    interaction.          setSpecificPair(pv1name, pv2name, pair);
    interactionWithStress.setSpecificPair(pv1name, pv2name, PairwiseStressWrapper<PairwiseInteraction>(pair));
}


template class InteractionPair_withStress<Pairwise_DPD>;
template class InteractionPair_withStress<Pairwise_LJ>;
template class InteractionPair_withStress<Pairwise_LJObjectAware>;
template class InteractionPair_withStress<Pairwise_MDPD>;
