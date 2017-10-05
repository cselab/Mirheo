#pragma once
#include "interface.h"
#include <core/xml/pugixml.hpp>

class InteractionRBCMembrane : public Interaction
{
	float epsilon, sigma;

public:
	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl1, CellList* cl2, const float t, cudaStream_t stream) override;

	InteractionRBCMembrane(pugi::xml_node node);

	~InteractionRBCMembrane() = default;
};
