#pragma once
#include "interface.h"

#include <core/datatypes.h>

/**
 * Implementation of short-range symmetric pairwise interactions
 */
template<class PairwiseInteraction>
class InteractionPair : public Interaction
{
public:
	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;

	InteractionPair(std::string name, float rc, PairwiseInteraction interaction) :
		Interaction(name, rc), interaction(interaction)
	{ }

	~InteractionPair() = default;

private:
	PairwiseInteraction interaction;
};
