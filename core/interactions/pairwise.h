#pragma once
#include "interface.h"

#include <core/datatypes.h>

template<class PairwiseInteraction>
class InteractionPair : public Interaction
{
	PairwiseInteraction interaction;

public:
	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;

	InteractionPair(std::string name, float rc, PairwiseInteraction interaction) :
		Interaction(name, rc), interaction(interaction)
	{ }

	~InteractionPair() = default;
};
