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
	enum class InteractionType { Regular, Halo };

	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream);

	void regular(ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;
	void halo   (ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;

	InteractionPair(std::string name, float rc) :
		Interaction(name, rc)
	{ }

	void createPairwise(std::string pv1name, std::string pv2name, PairwiseInteraction interaction);

	~InteractionPair() = default;

private:
	std::map< std::pair<std::string, std::string>, PairwiseInteraction > intMap;
};
