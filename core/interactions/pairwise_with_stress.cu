#include "pairwise_with_stress.h"

#include "pairwise_interactions/dpd.h"
#include "pairwise_interactions/lj.h"

/**
 * Implementation of short-range symmetric pairwise interactions
 */

template<class PairwiseInteraction>
void InteractionPair_withStress<PairwiseInteraction>::regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
{
	if (lastStressTime+stressPeriod <= t || lastStressTime == t)
	{
		debug("Executing interaction '%s' with stress", name.c_str());

		if (pv2lastStressTime[pv1] != t)
		{
			pv1->local()->extraPerParticle.getData<Stress>("stress")->clear(0);
			pv2lastStressTime[pv1] = t;
		}

		if (pv2lastStressTime[pv2] != t)
		{
			pv2->local()->extraPerParticle.getData<Stress>("stress")->clear(0);
			pv2lastStressTime[pv2] = t;
		}

		pairWithStress.regular(pv1, pv2, cl1, cl2, t, stream);
		lastStressTime = t;
	}
	else
		pair.regular(pv1, pv2, cl1, cl2, t, stream);
}

template<class PairwiseInteraction>
void InteractionPair_withStress<PairwiseInteraction>::halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream)
{
	if (lastStressTime+stressPeriod <= t || lastStressTime == t)
	{
		debug("Executing interaction '%s' with stress", name.c_str());

		if (pv2lastStressTime[pv1] != t)
		{
			pv1->local()->extraPerParticle.getData<Stress>("stress")->clear(0);
			pv2lastStressTime[pv1] = t;
		}

		if (pv2lastStressTime[pv2] != t)
		{
			pv2->local()->extraPerParticle.getData<Stress>("stress")->clear(0);
			pv2lastStressTime[pv2] = t;
		}

		pairWithStress.halo(pv1, pv2, cl1, cl2, t, stream);
		lastStressTime = t;
	}
	else
		pair.halo(pv1, pv2, cl1, cl2, t, stream);
}

template<class PairwiseInteraction>
void InteractionPair_withStress<PairwiseInteraction>::setPrerequisites(ParticleVector* pv1, ParticleVector* pv2)
{
	info("Interaction '%s' requires channel 'stress' from PVs '%s' and '%s'",
			name.c_str(), pv1->name.c_str(), pv2->name.c_str());

	pv1->requireDataPerParticle<Stress>("stress", false);
	pv2->requireDataPerParticle<Stress>("stress", false);

	pv2lastStressTime[pv1] = -1;
	pv2lastStressTime[pv2] = -1;
}

template<class PairwiseInteraction>
InteractionPair_withStress<PairwiseInteraction>::InteractionPair_withStress(std::string name, float rc, float stressPeriod) :

	Interaction(name, rc),
	stressPeriod(stressPeriod),
	pair(name, rc),
	pairWithStress(name, rc)
{ }

template<class PairwiseInteraction>
void InteractionPair_withStress<PairwiseInteraction>::createPairwise(std::string pv1name, std::string pv2name, PairwiseInteraction interaction)
{
	pair.          createPairwise(pv1name, pv2name, interaction);
	pairWithStress.createPairwise(pv1name, pv2name, PairwiseStressWrapper<PairwiseInteraction>(interaction));
}


template class InteractionPair_withStress<Pairwise_DPD>;
template class InteractionPair_withStress<Pairwise_LJ>;
