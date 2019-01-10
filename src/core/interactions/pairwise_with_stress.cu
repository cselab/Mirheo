#include "pairwise_with_stress.h"

#include "pairwise_interactions/dpd.h"
#include "pairwise_interactions/lj.h"
#include "pairwise_interactions/lj_object_aware.h"


template<class PairwiseInteraction>
InteractionPair_withStress<PairwiseInteraction>::InteractionPair_withStress(
    const YmrState *state, std::string name, std::string stressName, float rc, float stressPeriod, PairwiseInteraction pair) :

    Interaction(state, name, rc),
    stressName(stressName),
    stressPeriod(stressPeriod),
    interaction(state, name, rc, pair),
    interactionWithStress(state, name, rc, PairwiseStressWrapper<PairwiseInteraction>(stressName, pair))
{}

template<class PairwiseInteraction>
InteractionPair_withStress<PairwiseInteraction>::~InteractionPair_withStress() = default;

template<class PairwiseInteraction>
void InteractionPair_withStress<PairwiseInteraction>::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2)
{
    info("Interaction '%s' requires channel 'stress' from PVs '%s' and '%s'",
         name.c_str(), pv1->name.c_str(), pv2->name.c_str());

    pv1->requireDataPerParticle<Stress>(stressName, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);
    pv2->requireDataPerParticle<Stress>(stressName, ExtraDataManager::CommunicationMode::None, ExtraDataManager::PersistenceMode::None);

    pv2lastStressTime[pv1] = -1;
    pv2lastStressTime[pv2] = -1;
}

template<class PairwiseInteraction>
void InteractionPair_withStress<PairwiseInteraction>::regular(
        ParticleVector* pv1, ParticleVector* pv2,
        CellList* cl1, CellList* cl2, cudaStream_t stream)
{
    float t = state->currentTime;
    
    if (lastStressTime+stressPeriod <= t || lastStressTime == t)
    {
        debug("Executing interaction '%s' with stress", name.c_str());

        if (pv2lastStressTime[pv1] != t)
        {
            pv1->local()->extraPerParticle.getData<Stress>(stressName)->clear(0);
            pv2lastStressTime[pv1] = t;
        }

        if (pv2lastStressTime[pv2] != t)
        {
            pv2->local()->extraPerParticle.getData<Stress>(stressName)->clear(0);
            pv2lastStressTime[pv2] = t;
        }

        interactionWithStress.regular(pv1, pv2, cl1, cl2, stream);
        lastStressTime = t;
    }
    else
        interaction.regular(pv1, pv2, cl1, cl2, stream);
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
        {
            pv1->local()->extraPerParticle.getData<Stress>(stressName)->clear(0);
            pv2lastStressTime[pv1] = t;
        }

        if (pv2lastStressTime[pv2] != t)
        {
            pv2->local()->extraPerParticle.getData<Stress>(stressName)->clear(0);
            pv2lastStressTime[pv2] = t;
        }

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
    interactionWithStress.setSpecificPair(pv1name, pv2name, PairwiseStressWrapper<PairwiseInteraction>(stressName, pair));
}


template class InteractionPair_withStress<Pairwise_DPD>;
template class InteractionPair_withStress<Pairwise_LJ>;
template class InteractionPair_withStress<Pairwise_LJObjectAware>;
