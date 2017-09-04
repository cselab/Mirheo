#pragma once
#include "interface.h"
#include <core/xml/pugixml.hpp>

class InteractionLJ : public Interaction
{
	float epsilon, sigma;

public:
	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream);

	InteractionLJ(pugi::xml_node node);

	~InteractionLJ() = default;
};

class InteractionLJ_objectAware : public Interaction
{
	float epsilon, sigma;

public:
	void _compute(InteractionType type, ParticleVector* pv1, ParticleVector* pv2, CellList* cl, const float t, cudaStream_t stream);

	InteractionLJ_objectAware(pugi::xml_node node);

	~InteractionLJ_objectAware() = default;
};

