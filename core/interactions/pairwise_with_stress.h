#pragma once
#include "interface.h"
#include "pairwise.h"
#include "pairwise_interactions/stress_wrapper.h"

#include <core/datatypes.h>
#include <map>

/**
 * Implementation of short-range symmetric pairwise interactions
 */
template<class PairwiseInteraction>
class InteractionPair_withStress : public Interaction
{
public:
	enum class InteractionType { Regular, Halo };

	void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
	void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
	void setPrerequisites(ParticleVector* pv1, ParticleVector* pv2);

	InteractionPair_withStress(std::string name, float rc, float stressPeriod, PairwiseInteraction pair);

	void setSpecificPair(std::string pv1name, std::string pv2name, PairwiseInteraction pair);

	~InteractionPair_withStress() = default;

private:
	float stressPeriod;
	float lastStressTime{-1e6};

	std::map<ParticleVector*, float> pv2lastStressTime;

	InteractionPair<PairwiseInteraction> interaction;
	InteractionPair<PairwiseStressWrapper<PairwiseInteraction>> interactionWithStress;
};
